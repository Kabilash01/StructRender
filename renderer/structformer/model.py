"""StructFormer — KB §10 Milestone 5, ladder Model C.

Assembles the §3 *Runtime Neural Topology* path that exists at M5::

    structural buffers -> CNN STEM -> WINDOW TRANSFORMER ENCODER
                       -> WINDOW TRANSFORMER DECODER -> CNN HEAD -> styled RGB
                                    ^
                             cached style token

Deliberately absent at M5, each arriving at its own milestone: temporal warped
features (M6), entity masks / EntityKey / EPAM tokens / entity-masked
cross-attention (M7-M8), adversarial loss (optional, Decision 4). Frame-independent
by construction, so it is directly comparable to M4's U-Net on the §11 ladder.

Style conditioning (§4 *Style Conditioning Module*) enters as a **graph input**,
``style_token``, matching §5's Runtime Tensor Contract. The style encoder itself
is outside the graph and its output is cached until the style changes — §4's
"does not run a heavy text encoder every frame". Inside the model the token only
produces a FiLM scale/shift on the stem features; it cannot mutate anything else.

Shapes are fixed at construction (KB §12: "prefer fixed/bounded shapes"), and the
runtime contract is ``(1, 7, 288, 512) -> (1, 3, 288, 512)``, unchanged from M4.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from renderer.structformer.cnn_stem import IN_CHANNELS, STEM_STRIDE, CNNStem
from renderer.structformer.decoder import DecoderStage
from renderer.structformer.encoder import EncoderStage
from renderer.structformer.rgb_head import RGBHead

STYLE_DIM = 16


class StructFormer(nn.Module):
    """Structural buffers (+ cached style token) -> styled RGB, one frame at a time.

    Args:
        image_size: ``(H, W)``; both must be divisible by ``4 * window_size``.
        embed_dim: stage-1 width. Stage 2 is ``2 * embed_dim``.
        depths: blocks per (encoder stage 1, encoder stage 2, decoder stage).
        num_heads: heads per (stage 1, stage 2); an ``int`` applies to both. The
            decoder runs at stage-1 width and resolution, so it uses ``heads[0]``.
        window_sizes: window side at (stage-1, stage-2) resolution.
        attn_impl / shift_impl: see :mod:`renderer.structformer.window_attention`.

    The M5 export probe instantiates this at minimum capacity (``embed_dim=32``,
    one head, one block per stage). Capacity is a training question; the operator
    set, tensor ranks and shape flow are identical at any width.
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
        h, w = image_size
        if h % (STEM_STRIDE * window_sizes[0]) or w % (STEM_STRIDE * window_sizes[0]):
            raise ValueError(f"image_size {image_size} incompatible with window {window_sizes[0]}")
        res1 = (h // STEM_STRIDE, w // STEM_STRIDE)
        res2 = (res1[0] // 2, res1[1] // 2)
        if res2[0] % window_sizes[1] or res2[1] % window_sizes[1]:
            raise ValueError(f"stage-2 resolution {res2} incompatible with window {window_sizes[1]}")
        self.image_size = image_size
        self.style_dim = style_dim
        heads = (num_heads, num_heads) if isinstance(num_heads, int) else tuple(num_heads)
        if len(heads) != 2:
            raise ValueError(f"num_heads must be an int or a pair, got {num_heads!r}")

        self.stem = CNNStem(in_channels, embed_dim)
        # FiLM: one scale and one shift per stem channel, from the cached token.
        self.style_film = nn.Linear(style_dim, 2 * embed_dim)
        self.stage1 = EncoderStage(
            embed_dim, embed_dim * 2, res1, depths[0], heads[0], window_sizes[0],
            attn_impl, shift_impl,
        )
        self.stage2 = EncoderStage(
            embed_dim * 2, None, res2, depths[1], heads[1], window_sizes[1],
            attn_impl, shift_impl,
        )
        self.decoder = DecoderStage(
            embed_dim * 2, embed_dim, res1, depths[2], heads[0], window_sizes[0],
            attn_impl, shift_impl,
        )
        self.head = RGBHead(embed_dim, embed_dim, out_channels)

    def forward(self, structure: torch.Tensor, style_token: torch.Tensor) -> torch.Tensor:
        x = self.stem(structure)
        gamma, beta = self.style_film(style_token).chunk(2, dim=-1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x, skip1 = self.stage1(x)
        x, _ = self.stage2(x)
        x = self.decoder(x, skip1)
        return self.head(x)


class StructFormerSingleStyle(nn.Module):
    """:class:`StructFormer` with the M5 style token held as a learned constant.

    Exists so ladder Model C can run through the *same* training loop as Model A,
    whose contract is ``model(x) -> rgb``. :class:`StructFormer` is two-input by
    design (Decision 45, item 3: the exported graph is
    ``(structure, style_token) -> rgb``) and that stays true — this wrapper only
    supplies the token, it does not remove the input.

    The token is one learned ``[1, style_dim]`` parameter because M5 trains a
    **single** style (``oil_impasto``, Decision 40). §4's *Style Conditioning
    Module* has a style *encoder* produce the token from a prompt/reference and
    then caches it until the style changes; at one style the encoder's entire
    output is one cached vector, so learning that vector directly is the same
    tensor with the encoder deferred to the multi-style milestone. A fixed random
    or zero token would work too, but would leave ``style_film`` learning a plain
    bias and the style path untrained.

    Nothing style-conditioned is stored per entity — Decision 12's ``M_{i,s}``
    stays deferred, and EPAM does not exist yet at M5.
    """

    def __init__(self, **structformer_kwargs) -> None:
        super().__init__()
        self.net = StructFormer(**structformer_kwargs)
        # NOT zeros. A zero token makes d(loss)/d(style_film.weight) exactly zero,
        # so the FiLM projection is a dead parameter at initialisation — it only
        # starts learning after the token itself has drifted off zero. Caught by
        # test_gradient_reaches_every_parameter_and_is_non_zero, which checks
        # gradient magnitude rather than `is not None`. Same std as the
        # relative-position-bias table.
        self.style_token = nn.Parameter(torch.empty(1, self.net.style_dim))
        nn.init.trunc_normal_(self.style_token, std=0.02)

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        return self.net(structure, self.style_token.expand(structure.shape[0], -1))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
