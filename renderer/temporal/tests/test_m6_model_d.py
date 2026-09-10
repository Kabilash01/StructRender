"""Ladder Model D as it is actually configured — KB §10 Milestone 6.

Everything here builds the model from `configs/temporal/m6_temporal.json` rather
than from constructor defaults, so the assertions bind the *shipped* architecture.
Pure CPU and synthetic tensors: no GPU, no recorded session, no teacher targets,
no checkpoint.

Companion to `test_m6_config.py` (the config's non-architecture blocks against
Model C's) and `test_m6_temporal_loss.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from renderer.structformer.model import StructFormer, StructFormerSingleStyle, parameter_count
from renderer.temporal.temporal_model import TemporalStructFormer
from renderer.training.dataset import CHANNEL_NAMES, build_input
from renderer.training.train_baseline import load_config
from renderer.training.train_temporal import build_model, model_kwargs

from .conftest import ROOT

CONFIG = ROOT / "configs" / "temporal" / "m6_temporal.json"

#: Model C's parameter count, pinned in `renderer/structformer/tests/test_m5_model.py`
#: and recomputed below rather than trusted.
MODEL_C_PARAMETERS = 7_775_467

#: Model A's, likewise — Decision 47's two-sided band is stated against it.
MODEL_A_PARAMETERS = 7_853_763

#: Model D's, MEASURED. The delta over Model C is the 1x1 fusion convolution and
#: nothing else; pinned so an architecture edit fails here instead of silently
#: changing what the §11 ladder compares.
MODEL_D_PARAMETERS = 8_236_747

#: Constraints: stem /4 then stage 2 halves again, so H/4 % 8 == 0 (stage-1
#: window 8) and H/8 % 4 == 0 (stage-2 window 4).
SMALL_HW = (64, 64)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def model_d(cfg) -> TemporalStructFormer:
    torch.manual_seed(0)
    return build_model(cfg).eval()


def _small(cfg) -> TemporalStructFormer:
    small = {**cfg, "model": {**cfg["model"], "image_size": list(SMALL_HW)}}
    torch.manual_seed(0)
    return build_model(small)


def _inputs(bs: int, hw=SMALL_HW, mask_value=None):
    """Structural pair + a plausible motion field + a mask."""
    h, w = hw
    motion = (torch.rand(bs, 2, h, w) - 0.5) * 0.02
    mask = torch.rand(bs, 1, h, w) if mask_value is None else torch.full((bs, 1, h, w), mask_value)
    return torch.rand(bs, 7, h, w), torch.rand(bs, 7, h, w), motion, mask


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def test_forward_shape_is_the_runtime_tensor_contract(model_d):
    """KB §5 Runtime Tensor Contract — the RGB contract is Model C's exactly, so
    the two rungs stay directly comparable on the §11 ladder."""
    s_prev, s, motion, mask = _inputs(1, (288, 512))
    with torch.no_grad():
        prev_features, _ = model_d.encode(s_prev)
        rgb = model_d(s, prev_features, motion, mask)
    assert prev_features.shape == (1, 480, 36, 64)
    assert rgb.shape == (1, 3, 288, 512)


def test_forward_shape_holds_for_a_batch(cfg):
    m = _small(cfg).eval()
    s_prev, s, motion, mask = _inputs(3)
    with torch.no_grad():
        rgb = m(s, m.encode(s_prev)[0], motion, mask)
    assert rgb.shape == (3, 3, *SMALL_HW)


def test_output_range_is_model_c_s(model_d):
    """Sigmoid head, [0, 1] — the same output contract as Models A and C."""
    s_prev, s, motion, mask = _inputs(1, (288, 512))
    with torch.no_grad():
        rgb = model_d(s, model_d.encode(s_prev)[0], motion, mask)
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0


def test_output_range_holds_on_extreme_inputs_and_extreme_motion(cfg):
    """Real inputs carry a background sentinel (depth_inv exactly 0) and zeroed
    background normals; real motion fields can point well outside the frame,
    where `grid_sample` returns zero-padding."""
    m = _small(cfg).eval()
    h, w = SMALL_HW
    for name, x in (("zeros", torch.zeros(2, 7, h, w)),
                    ("ones", torch.ones(2, 7, h, w)),
                    ("large", torch.full((2, 7, h, w), 1e3))):
        for mname, motion in (("still", torch.zeros(2, 2, h, w)),
                              ("off_frame", torch.full((2, 2, h, w), 5.0))):
            with torch.no_grad():
                rgb = m(x, m.encode(x)[0], motion, torch.ones(2, 1, h, w))
            assert torch.isfinite(rgb).all(), f"{name}/{mname} produced a non-finite output"
            assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0, f"{name}/{mname}"


def test_the_cacheable_state_is_the_pre_fusion_encoder_output(cfg):
    """The runtime carries ONE tensor across frames, and it must be `encode()`'s
    output — a function of THIS frame's structural buffers alone.

    That is the previous-feature decision, made checkable: because the state does
    not depend on whether the frame that produced it had history of its own,
    training (which recomputes it from S_{t-1}) and inference (which cached it a
    frame ago) compute the identical tensor, and there is no exposure gap to
    trade off. A post-fusion state would fail this and would make the model an
    unbounded recurrence instead.
    """
    m = _small(cfg)
    torch.nn.init.normal_(m.temporal_fuse.weight, std=0.5)
    m.eval()
    s_prev, s, motion, mask = _inputs(2, mask_value=1.0)
    with torch.no_grad():
        cold, _ = m.encode(s)
        # the same frame, this time rendered WITH history
        rgb = m(s, m.encode(s_prev)[0], motion, mask)
        warm, _ = m.encode(s)
    assert torch.equal(cold, warm)
    assert rgb.shape == (2, 3, *SMALL_HW)


# --------------------------------------------------------------------------- #
# the fusion's §4 boundaries, measured rather than asserted from the source
# --------------------------------------------------------------------------- #
def test_zero_confidence_reduces_to_model_c(cfg):
    """§4: "invalid warped history is discarded", "newly disoccluded pixels are
    not filled from stale history". Both must hold for ANY weights the fusion
    could learn, so the fusion is trained away from zero first and the check is
    then made against a real Model C carrying the same weights.

    This is also the drift guard: Model D re-implements Model C's stem -> FiLM ->
    stage1 -> stage2 -> decoder -> head path in `encode`/`decode` instead of
    calling `StructFormer.forward`, and if that path ever diverges from Model C's
    this equality is what fails.
    """
    m = _small(cfg)
    torch.nn.init.normal_(m.temporal_fuse.weight, std=0.5)
    torch.nn.init.normal_(m.temporal_fuse.bias, std=0.5)
    m.eval()

    c = StructFormer(**{**model_kwargs({**cfg, "model": {**cfg["model"],
                                                         "image_size": list(SMALL_HW)}}),
                        "image_size": SMALL_HW})
    c.load_state_dict({k[len("net."):]: v for k, v in m.state_dict().items()
                       if k.startswith("net.")})
    c.eval()

    s_prev, s, motion, _ = _inputs(2)
    with torch.no_grad():
        d_cold = m(s, m.encode(s_prev)[0], motion, torch.zeros(2, 1, *SMALL_HW))
        d_none = m(s)
        c_out = c(s, m.style_token.expand(2, -1))
    assert torch.equal(d_cold, c_out), "M_t = 0 must reproduce Model C exactly"
    assert torch.equal(d_none, c_out), "no history must reproduce Model C exactly"


def test_history_changes_the_output_where_confidence_is_high(cfg):
    """The complement of the test above: with a trained-away-from-zero fusion and
    a confident mask, history MUST reach the image. Without this, the test above
    would pass for a model whose temporal path is dead."""
    m = _small(cfg)
    torch.nn.init.normal_(m.temporal_fuse.weight, std=0.5)
    m.eval()
    s_prev, s, motion, _ = _inputs(2, mask_value=1.0)
    ones = torch.ones(2, 1, *SMALL_HW)
    with torch.no_grad():
        a = m(s, m.encode(s_prev)[0], motion, ones)
        b = m(s, m.encode(torch.rand_like(s_prev))[0], motion, ones)
    assert not torch.allclose(a, b), "two different histories produced the same frame"


def test_the_temporal_contribution_is_a_gated_residual_on_current_features(cfg):
    """§4: "temporal fusion never overrides current engine structure", and
    "invalid warped history is discarded".

    The fusion is recomposed here from its own parts — warp, input gate, 1x1 conv,
    output gate — and compared against `fuse()`. That is what makes the check
    non-vacuous: it pins the *shape* of the combination (a residual whose
    correction is multiplied by `M_t` **after** the convolution), not merely that
    some number came out. Substituting a blend, or moving the gate inside the
    convolution where its bias escapes it, fails here.
    """
    import torch.nn.functional as F

    from renderer.temporal.warp import resize_motion

    m = _small(cfg)
    torch.nn.init.normal_(m.temporal_fuse.weight, std=0.5)
    torch.nn.init.normal_(m.temporal_fuse.bias, std=0.5)
    s_prev, s, motion, mask = _inputs(2)
    with torch.no_grad():
        feats, _ = m.encode(s)
        prev, _ = m.encode(s_prev)
        fused = m.fuse(feats, prev, motion, mask)

        m_lr = F.adaptive_avg_pool2d(mask, m.feature_size)
        warped = m.warp(prev, resize_motion(motion, m.feature_size)) * m_lr
        delta = m.temporal_fuse(torch.cat((feats, warped), dim=1))
    assert torch.equal(fused, feats + m_lr * delta)

    # ...and the gate is exact, bias included: no confidence, no correction.
    with torch.no_grad():
        assert torch.equal(m.fuse(feats, prev, motion, torch.zeros_like(mask)), feats)


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
def test_gradient_reaches_every_parameter_and_is_non_zero(cfg):
    """KB §11 Required unit test, "gradient flow" — on MAGNITUDE, not `is not None`.

    A parameter can receive an all-zero gradient and be dead. `temporal_fuse` is
    zero-initialised, which is exactly the situation where a weaker check would
    pass while the new module learned nothing, so it is the parameter this test
    exists for.
    """
    m = _small(cfg)
    m.train()
    s_prev, s, motion, mask = _inputs(2)
    out = m(s, m.encode(s_prev)[0], motion, mask)
    torch.nn.functional.l1_loss(out, torch.rand(2, 3, *SMALL_HW)).backward()

    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"{len(missing)} parameters got no gradient at all: {missing[:8]}"
    dead = [n for n, p in m.named_parameters() if float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"{len(dead)} parameters got an all-zero gradient: {dead[:8]}"


def test_the_style_token_is_trained_not_decorative(cfg):
    m = _small(cfg)
    s_prev, s, motion, mask = _inputs(2)
    out = m(s, m.encode(s_prev)[0], motion, mask)
    torch.nn.functional.l1_loss(out, torch.rand(2, 3, *SMALL_HW)).backward()
    assert float(m.style_token.grad.abs().sum()) > 0.0
    assert m.style_token.shape == (1, cfg["model"]["style_dim"])


# --------------------------------------------------------------------------- #
# input channels — the ladder controls (Decisions 8 and 9)
# --------------------------------------------------------------------------- #
def test_model_d_takes_exactly_seven_input_channels(cfg, model_d):
    """Decision 35's 7-channel contract survives the temporal rung. Read off the
    stem's first convolution, not off the config."""
    assert cfg["model"]["in_channels"] == 7
    assert model_d.net.stem.body[0][0].in_channels == 7
    assert len(CHANNEL_NAMES) == 7
    m = _small(cfg).eval()
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            m(torch.rand(1, 8, *SMALL_HW))


def test_motion_reaches_the_output_only_through_the_warp(cfg):
    """KB Decision 8 / §4 CNN Stem: "motion is not treated as an unexplained
    appearance channel". At M6 motion finally enters the model — as the argument
    of a warp and nothing else.

    Measured, not asserted from the source: with `M_t = 0` the warp's result is
    gated to nothing, so if motion had ANY other route into the network the
    output would move when the field is replaced. It does not.
    """
    m = _small(cfg)
    torch.nn.init.normal_(m.temporal_fuse.weight, std=0.5)
    torch.nn.init.normal_(m.temporal_fuse.bias, std=0.5)
    m.eval()
    s_prev, s, motion, _ = _inputs(2)
    zeros = torch.zeros(2, 1, *SMALL_HW)
    with torch.no_grad():
        prev = m.encode(s_prev)[0]
        a = m(s, prev, motion, zeros)
        b = m(s, prev, torch.randn_like(motion) * 0.1, zeros)
    assert torch.equal(a, b)
    assert "motion" not in CHANNEL_NAMES


def test_entity_id_is_not_an_input_channel_for_model_d():
    """KB Decision 9 — IDs are addresses. Model D eats the same `build_input`
    tensor as Models A and C, so the assertion is re-run against Model D's own
    contract rather than assumed to carry over. The entity-ID map DOES reach M6,
    but only inside the confidence mask's S7 hard zero, where it is used as an
    address (which entities have no previous transform), never as a value."""
    from data_pipeline.teacher.adapter import FrameStructure

    rng = np.random.default_rng(0)
    fs = FrameStructure(
        session_id="synthetic", frame_id=0,
        depth=rng.random((16, 16)).astype(np.float32),
        normal=rng.random((16, 16, 3)).astype(np.float32),
        albedo=rng.random((16, 16, 3)).astype(np.float32),
        entity_id=rng.integers(0, 9, (16, 16)).astype(np.int32),
    )
    scrambled = FrameStructure(
        session_id=fs.session_id, frame_id=fs.frame_id, depth=fs.depth,
        normal=fs.normal, albedo=fs.albedo,
        entity_id=(fs.entity_id * 7919 + 13).astype(np.int32),
    )
    assert np.array_equal(build_input(fs), build_input(scrambled))


def test_m6_does_not_wire_in_entity_or_epam_machinery(model_d):
    """§11 places Model D at Temporal: YES / Structured Entity Use: No / EPAM: No.
    `masked_entity_attention` stays probe-only until M7."""
    import ast
    import inspect

    from renderer.structformer import masked_entity_attention as probe_only
    from renderer.temporal import temporal_model as mod

    tree = ast.parse(inspect.getsource(mod))
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any("masked_entity_attention" in m for m in imported), imported
    assert not any("epam" in m.lower() for m in imported), imported

    probe_classes = tuple(v for v in vars(probe_only).values()
                          if isinstance(v, type) and issubclass(v, torch.nn.Module))
    leaked = [n for n, mm in model_d.named_modules() if isinstance(mm, probe_classes)]
    assert not leaked, f"probe-only entity modules reached Model D: {leaked}"


# --------------------------------------------------------------------------- #
# capacity
# --------------------------------------------------------------------------- #
def test_model_d_capacity_against_model_c(cfg, model_d):
    """The §11 ladder isolates architecture, so capacity is the confound, and
    Decision 47's band is two-sided because an undershoot confounds as badly as
    an overshoot. All three counts are RECOMPUTED here, never quoted."""
    c = parameter_count(StructFormerSingleStyle(**model_kwargs(cfg)))
    d = parameter_count(model_d)
    assert c == MODEL_C_PARAMETERS, f"Model C changed: {c}"
    assert d == MODEL_D_PARAMETERS, (
        f"Model D's parameter count moved to {d} ({d / c:.4f}x Model C). Update the pin "
        "deliberately — the ladder comparison depends on this number being stated."
    )
    assert d - c == parameter_count(model_d.temporal_fuse), (
        "Model D must differ from Model C by the fusion convolution and nothing else"
    )
    assert 0.9 <= d / c <= 1.1, f"Model D is {d / c:.4f}x Model C, outside Decision 47's band"
    assert 0.9 <= d / MODEL_A_PARAMETERS <= 1.1, f"Model D is {d / MODEL_A_PARAMETERS:.4f}x Model A"


def test_architecture_matches_the_config(cfg, model_d):
    """Every architecture number in the config reaches the built model."""
    m = cfg["model"]
    net = model_d.net
    assert net.image_size == tuple(m["image_size"])
    assert net.stem.body[0][0].out_channels == m["embed_dim"]
    assert [len(net.stage1.blocks), len(net.stage2.blocks), len(net.decoder.blocks)] == m["depths"]
    assert net.stage1.blocks[0].attn.num_heads == m["num_heads"][0]
    assert net.stage2.blocks[0].attn.num_heads == m["num_heads"][1]
    assert net.decoder.blocks[0].attn.num_heads == m["num_heads"][0]
    assert net.stage1.blocks[0].window_size == m["window_sizes"][0]
    assert net.stage2.blocks[0].window_size == m["window_sizes"][1]
    assert net.stage1.blocks[0].attn.attn_impl == m["attn_impl"]
    assert net.stage1.blocks[1].shift_impl == m["shift_impl"]
    # The fusion sits at the encoder/decoder junction §3's topology puts it at.
    assert model_d.feature_dim == 2 * m["embed_dim"]
    assert model_d.feature_size == (m["image_size"][0] // 8, m["image_size"][1] // 8)
    assert model_d.temporal_fuse.kernel_size == (1, 1)


def test_every_stage_has_a_shifted_block(model_d):
    for name, stage in (("stage1", model_d.net.stage1), ("stage2", model_d.net.stage2),
                        ("decoder", model_d.net.decoder)):
        shifts = [b.shift_size for b in stage.blocks]
        assert any(s > 0 for s in shifts), f"{name} has no shifted block: {shifts}"
