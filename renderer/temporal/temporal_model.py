"""Temporal StructFormer — ladder **Model D**, KB §10 Milestone 6.

Model C (`renderer/structformer/model.py`) with one thing added: the previous
frame's **encoder features** are reprojected into the current frame by the
engine's motion field and fused under the §4 confidence mask.

**Where the fusion goes is not a free choice.** KB §3's *Runtime Neural Topology*
puts "Temporal warped features" as a sibling of "Current spatial features" on the
branch **between the encoder and the decoder**::

              WINDOW TRANSFORMER ENCODER
                          |
        +-----------------+------------------+
        |                 |                  |
   Current spatial   Temporal warped    Entity masks  (M7/M8, absent here)
      features          features
        +-----------------+------------------+
                          |
              WINDOW TRANSFORMER DECODER

so the fused tensor is Model C's stage-2 output — ``(B, 2*embed_dim, H/8, W/8)``
— and everything upstream and downstream of that junction is Model C, untouched.
§4 specifies the warp on **features**, not on output pixels, and that is what
happens here; nothing warps RGB.

**The fusion, and why it is shaped this way.** ::

    warped = W(F_{t-1}, V_t) * M_t          # invalid history is ZEROED, not discouraged
    delta  = conv1x1([F_t, warped])
    F_t'   = F_t + M_t * delta              # exactly zero contribution where M_t = 0

Three §4 boundaries drive it, and each is structural rather than hoped-for:

* *"invalid warped history is discarded"* and *"newly disoccluded pixels are not
  filled from stale history"* — the **output** gate makes the contribution
  identically ``0`` wherever ``M_t`` is 0, for any weights the conv could ever
  learn. The input gate is redundant for that (a 1x1 conv mixes no neighbours)
  and is kept anyway so the convolution never *reads* a stale value: KB Decision
  52 records what happened the last time a mask only discouraged instead of
  excluding.
* *"temporal fusion never overrides current engine structure"* — the fusion is a
  **residual** on ``F_t``, so the current frame's structural features are always
  present at full strength; history can only add a correction.

``temporal_fuse`` is **zero-initialised**, so an untrained Model D is numerically
identical to Model C. Asserted by ``test_zero_confidence_reduces_to_model_c``.

That property was the reason ``train.init_from`` (KB Decision 19 / §4 *Training
Curriculum Controller*: "stages reuse checkpoints") was believed to be a true
continuation. **KB Decision 56 disproved it**: the perturbation does not come
from the fusion conv but from AdamW's first bias-corrected step, whose magnitude
is ~``lr`` regardless of gradient scale, so resuming at ``lr = 2e-4`` with no
warmup moves Model C's converged solution an order of magnitude worse in two
steps. It is a warm restart, not a fine-tune. Model D therefore trains from
scratch and ``init_from`` is ``null`` in the shipped config.

**Motion is still not an appearance channel** (KB Decision 8, §4 *CNN Stem*). The
stem takes the same seven channels as Models A and C; ``motion`` enters this
module only as the argument of a warp, and with ``confidence = 0`` the output is
bit-identical for any motion field — which is what
``test_motion_reaches_the_output_only_through_the_warp`` measures rather than
asserts from the source.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from renderer.structformer.cnn_stem import IN_CHANNELS, STEM_STRIDE
from renderer.structformer.model import STYLE_DIM, StructFormer
from renderer.temporal.warp import EngineGuidedWarp, resize_motion


class TemporalStructFormer(nn.Module):
    """Model D. Structural buffers + engine motion + confidence -> styled RGB.

    Constructor arguments are Model C's, plus nothing: the architecture is
    identical and the only new parameters are ``temporal_fuse``. The style token
    is held as a learned constant exactly as
    :class:`~renderer.structformer.model.StructFormerSingleStyle` does — M6
    trains the same single style (``oil_impasto``) as M5, and making the token a
    graph input here would be a second difference between the rungs.

    ``model(x)`` IS a valid call — the temporal arguments default to ``None`` because
    cold start (frame 0, or any frame with no usable previous buffer) legitimately
    needs it, and it reduces the model exactly to Model C. An earlier version of
    this docstring claimed the signature prevented frame-independent use; it does
    not, and the guard lives in `train_baseline` refusing this arch instead
    (KB Decision 58).
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (288, 512),
        in_channels: int = IN_CHANNELS,
        out_channels: int = 3,
        embed_dim: int = 32,
        depths: tuple[int, int, int] = (1, 1, 1),
        num_heads: int | tuple[int, int] = 1,
        window_sizes: tuple[int, int] = (8, 4),
        style_dim: int = STYLE_DIM,
        attn_impl: str = "manual",
        shift_impl: str = "roll",
    ) -> None:
        super().__init__()
        self.net = StructFormer(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_sizes=window_sizes,
            style_dim=style_dim,
            attn_impl=attn_impl,
            shift_impl=shift_impl,
        )
        # Same initialisation as StructFormerSingleStyle: NOT zeros, or
        # `style_film` starts with an exactly-zero gradient and is dead.
        self.style_token = nn.Parameter(torch.empty(1, self.net.style_dim))
        nn.init.trunc_normal_(self.style_token, std=0.02)

        h, w = image_size
        # The encoder/decoder junction: stem stride 4, then one stride-2 stage.
        self.feature_size = (h // (STEM_STRIDE * 2), w // (STEM_STRIDE * 2))
        self.feature_dim = embed_dim * 2
        self.warp = EngineGuidedWarp(*self.feature_size)
        self.temporal_fuse = nn.Conv2d(2 * self.feature_dim, self.feature_dim, 1)
        nn.init.zeros_(self.temporal_fuse.weight)
        nn.init.zeros_(self.temporal_fuse.bias)

    # ------------------------------------------------------------------ #
    # the three pieces, so the runtime can cache the middle one
    # ------------------------------------------------------------------ #
    def encode(self, structure: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Model C's forward up to the encoder/decoder junction.

        Returns ``(encoder_features, skip)``. ``encoder_features`` is the tensor
        that is cached and reprojected on the next frame; it is a function of
        **this frame's structural buffers alone**, so the runtime can compute it
        once per frame and the training loop computes the identical thing. See
        the module-level note on the previous-feature decision.
        """
        x = self.net.stem(structure)
        gamma, beta = self.net.style_film(
            self.style_token.expand(structure.shape[0], -1)
        ).chunk(2, dim=-1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x, skip = self.net.stage1(x)
        x, _ = self.net.stage2(x)
        return x, skip

    def fuse(
        self,
        features: torch.Tensor,
        prev_features: torch.Tensor,
        motion: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """``F_t, F_{t-1}, V_t, M_t -> F_t'`` at feature resolution.

        ``motion`` and ``confidence`` come in at **image** resolution, because
        that is the resolution the dataset, the temporal loss and the M6 metric
        all speak. Motion needs no value rescaling on the way down — screen-UV is
        resolution-independent (KB Decision 25), which is exactly the property
        :func:`~renderer.temporal.warp.resize_motion` preserves.

        The mask is reduced with **area** averaging, not bilinear: one feature
        cell covers an 8x8 pixel block, so "half of my footprint has no usable
        history" should read as half confidence. Bilinear would sample a single
        interior point and could report full confidence for a cell that is mostly
        disoccluded.
        """
        m = F.adaptive_avg_pool2d(confidence, self.feature_size)
        v = resize_motion(motion, self.feature_size)
        warped = self.warp(prev_features, v) * m
        delta = self.temporal_fuse(torch.cat((features, warped), dim=1))
        return features + m * delta

    def decode(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.net.head(self.net.decoder(features, skip))

    # ------------------------------------------------------------------ #
    def forward(
        self,
        structure: torch.Tensor,
        prev_features: torch.Tensor | None = None,
        motion: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Styled RGB for one frame.

        ``prev_features`` is the :meth:`encode` output of frame ``t-1``. Passing
        ``None`` is the **cold start** — frame 0, or any frame whose previous
        buffers do not exist — and reduces the model exactly to Model C.
        """
        features, skip = self.encode(structure)
        if prev_features is not None:
            if motion is None or confidence is None:
                raise ValueError("prev_features requires motion and confidence")
            features = self.fuse(features, prev_features, motion, confidence)
        return self.decode(features, skip)
