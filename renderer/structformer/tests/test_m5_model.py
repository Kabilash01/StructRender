"""Ladder Model C as it is actually configured — KB §10 Milestone 5.

Everything here builds the model from `configs/structformer/m5_structformer.json`
rather than from constructor defaults, so the assertions bind the *shipped*
architecture and not a hypothetical one. Pure CPU and synthetic tensors: no GPU,
no recorded session, no teacher targets, no checkpoint.

Companion to `test_m5_config.py`, which checks the config's non-architecture
blocks against Model A's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from renderer.baselines.unet import UNetBaseline
from renderer.structformer.model import StructFormerSingleStyle, parameter_count
from renderer.training.dataset import CHANNEL_NAMES, build_input
from renderer.training.train_baseline import build_model, load_config

from .conftest import ROOT

CONFIG = ROOT / "configs" / "structformer" / "m5_structformer.json"

#: Model A's measured parameter count, from `renderer/baselines/unet.py` at
#: `base_channels=32`. Recomputed in `test_model_c_capacity_against_model_a`
#: rather than trusted, so a change to either model shows up as a failure.
MODEL_A_PARAMETERS = 7_853_763

#: Model C's parameter count at the configured architecture, MEASURED not chosen.
#: Pinned so that any edit to embed_dim / depths / heads / mlp_ratio fails here
#: instead of silently changing what the ladder compares.
MODEL_C_PARAMETERS = 7_775_467

#: Small enough to be fast on CPU while keeping every parameter tensor live.
#: Constraints: the stem downsamples 4x and stage 2 halves again, so H and W must
#: satisfy H/4 % 8 == 0 (stage-1 window 8) and H/8 % 4 == 0 (stage-2 window 4).
SMALL_HW = (64, 64)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def model_c(cfg) -> StructFormerSingleStyle:
    torch.manual_seed(0)
    return build_model(cfg).eval()


def _small(cfg) -> StructFormerSingleStyle:
    """The configured architecture at a small image size — same parameters."""
    small = {**cfg, "model": {**cfg["model"], "image_size": list(SMALL_HW)}}
    torch.manual_seed(0)
    return build_model(small)


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def test_forward_shape_is_the_runtime_tensor_contract(model_c):
    """KB §5 Runtime Tensor Contract, and Decision 35's export shapes — identical
    to Model A's, so the two rungs are directly comparable on the §11 ladder."""
    with torch.no_grad():
        rgb = model_c(torch.rand(1, 7, 288, 512))
    assert rgb.shape == (1, 3, 288, 512)


def test_forward_shape_holds_for_a_batch(cfg):
    """The training loop runs batch 8; the shift mask and window partition are
    both batch-dependent index arithmetic, so batch 1 passing proves little."""
    m = _small(cfg).eval()
    with torch.no_grad():
        rgb = m(torch.rand(3, 7, *SMALL_HW))
    assert rgb.shape == (3, 3, *SMALL_HW)


def test_output_range_matches_model_a(model_c):
    """Sigmoid head: [0, 1], the same output contract as the U-Net, matching the
    8-bit sRGB teacher targets scaled to [0, 1]."""
    with torch.no_grad():
        rgb = model_c(torch.rand(1, 7, 288, 512))
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0


def test_output_range_holds_on_extreme_inputs(cfg):
    """Real inputs carry a background sentinel (depth_inv exactly 0) and zeroed
    background normals; LayerNorm on a constant window is the risk here."""
    m = _small(cfg).eval()
    for name, x in (
        ("zeros", torch.zeros(2, 7, *SMALL_HW)),
        ("ones", torch.ones(2, 7, *SMALL_HW)),
        ("large", torch.full((2, 7, *SMALL_HW), 1e3)),
    ):
        with torch.no_grad():
            rgb = m(x)
        assert torch.isfinite(rgb).all(), f"{name} input produced a non-finite output"
        assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0, name


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
def test_gradient_reaches_every_parameter_and_is_non_zero(cfg):
    """KB §11 Required unit test, "gradient flow" — strengthened past `is not None`.

    A parameter can receive a grad tensor that is all zeros and be dead: a
    relative-position-bias table that is never gathered, or an attention branch
    whose output is multiplied out. `is not None` passes for all of those. The
    check is therefore on the magnitude, and it names every offender rather than
    the first, so one run says how much of the model is dead.
    """
    m = _small(cfg)
    m.train()
    out = m(torch.rand(2, 7, *SMALL_HW))
    torch.nn.functional.l1_loss(out, torch.rand(2, 3, *SMALL_HW)).backward()

    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"{len(missing)} parameters got no gradient at all: {missing[:8]}"
    dead = [n for n, p in m.named_parameters() if float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"{len(dead)} parameters got an all-zero gradient: {dead[:8]}"


def test_the_style_token_is_trained_not_decorative(cfg):
    """§4's Style Conditioning Module must actually reach the image. If the token
    carried no gradient it would be a dead input that the exported graph still
    binds, and M5's "style conditioning" deliverable would be a no-op."""
    m = _small(cfg)
    out = m(torch.rand(2, 7, *SMALL_HW))
    torch.nn.functional.l1_loss(out, torch.rand(2, 3, *SMALL_HW)).backward()
    assert float(m.style_token.grad.abs().sum()) > 0.0
    assert m.style_token.shape == (1, cfg["model"]["style_dim"])


# --------------------------------------------------------------------------- #
# input channels — the ladder controls (Decisions 8 and 9)
# --------------------------------------------------------------------------- #
def test_model_c_takes_exactly_seven_input_channels(cfg, model_c):
    """Decision 35's 7-channel contract holds for Model C too. Read off the stem's
    first convolution, not off the config, so config and code cannot disagree."""
    assert cfg["model"]["in_channels"] == 7
    assert model_c.net.stem.body[0][0].in_channels == 7
    assert len(CHANNEL_NAMES) == 7
    with pytest.raises(RuntimeError):
        m = _small(cfg).eval()
        with torch.no_grad():
            m(torch.rand(1, 8, *SMALL_HW))


def test_entity_id_is_not_an_input_channel_for_model_c():
    """KB Decision 9 — IDs are addresses. §11's Model N is the control for raw-ID
    input; leaking IDs into Model C destroys it exactly as it would in Model A.
    Model C eats the same `build_input` tensor, so the M4 assertion is re-run here
    against Model C's own contract rather than assumed to carry over."""
    from data_pipeline.teacher.adapter import FrameStructure

    rng = np.random.default_rng(0)
    fs = FrameStructure(
        session_id="synthetic",
        frame_id=0,
        depth=rng.random((16, 16)).astype(np.float32),
        normal=rng.random((16, 16, 3)).astype(np.float32),
        albedo=rng.random((16, 16, 3)).astype(np.float32),
        entity_id=rng.integers(0, 9, (16, 16)).astype(np.int32),
    )
    scrambled = FrameStructure(
        session_id=fs.session_id,
        frame_id=fs.frame_id,
        depth=fs.depth,
        normal=fs.normal,
        albedo=fs.albedo,
        entity_id=(fs.entity_id * 7919 + 13).astype(np.int32),
    )
    assert np.array_equal(build_input(fs), build_input(scrambled))


def test_motion_is_not_an_input_channel_for_model_c():
    """KB §4 CNN Stem: "motion is not treated as an unexplained appearance
    channel". Motion has an explicit warp operation and arrives at M6 (Decision 8),
    so no M5 module may read it."""
    import inspect

    from renderer.structformer import cnn_stem, decoder, encoder, model, rgb_head

    for mod in (cnn_stem, encoder, decoder, rgb_head, model):
        src = inspect.getsource(mod)
        assert '"motion"' not in src and "'motion'" not in src, mod.__name__
    assert "motion" not in CHANNEL_NAMES


def test_m5_does_not_wire_in_temporal_entity_or_epam_machinery(model_c):
    """§11 places Model C at Temporal: No / Structured Entity Use: No — it must
    differ from Model A in architecture ALONE. `masked_entity_attention` was
    written for the M5 export probe and stays probe-only until M7.

    Checked structurally, on imports and on the built module tree. A substring
    scan of the source was the first attempt and it failed on the model's own
    docstring, which names EPAM precisely to say it is absent — a test that
    forbids describing the boundary is a worse test.
    """
    import ast
    import inspect

    from renderer.structformer import masked_entity_attention as probe_only
    from renderer.structformer import model as model_mod

    tree = ast.parse(inspect.getsource(model_mod))
    imported = {
        n.module or ""
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
    } | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any("masked_entity_attention" in m for m in imported), imported

    probe_classes = tuple(
        v for v in vars(probe_only).values()
        if isinstance(v, type) and issubclass(v, torch.nn.Module)
    )
    leaked = [n for n, m in model_c.named_modules() if isinstance(m, probe_classes)]
    assert not leaked, f"probe-only entity modules reached Model C: {leaked}"


# --------------------------------------------------------------------------- #
# capacity
# --------------------------------------------------------------------------- #
def test_model_c_capacity_against_model_a(model_c):
    """The ladder is meant to measure architecture, so capacity is the confound.

    Both counts are RECOMPUTED here. Model C was first built at embed 96, which
    measured 0.160x Model A — a 6x capacity deficit under a stated parity
    rationale, and one that a one-sided ">1.5x" stop condition could not catch.
    Width is now 240 for parity (KB Decision 47), so the assertions below are
    two-sided: an undershoot is exactly as confounding as an overshoot, and it is
    the direction that makes a transformer look bad for a reason that is not its
    architecture.
    """
    a = parameter_count(UNetBaseline(in_channels=7, out_channels=3, base_channels=32))
    b = parameter_count(model_c)
    assert a == MODEL_A_PARAMETERS, f"Model A changed: {a}"
    assert b == MODEL_C_PARAMETERS, (
        f"Model C's parameter count moved to {b} ({b / a:.3f}x Model A). Update the pin "
        "deliberately — the ladder comparison depends on this number being stated."
    )
    # Parity is the POINT of the pin, not a coincidence of it (KB Decision 47).
    # The raw pin above catches any drift; this catches drift that happens to keep
    # the pin updated but abandons the control the ladder depends on.
    assert 0.9 <= b / a <= 1.1, (
        f"Model C is {b / a:.3f}x Model A. The §11 ladder is meant to isolate "
        "architecture, so capacity parity is a control, not a detail: outside this "
        "band a loss cannot be attributed to the architecture rather than the size."
    )



def test_architecture_matches_the_config(cfg, model_c):
    """Every architecture number in the config reaches the built model. Without
    this the config is documentation rather than the source of truth."""
    m = cfg["model"]
    net = model_c.net
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


def test_every_stage_has_a_shifted_block(model_c):
    """Depth 1 emits no shifted block, making window boundaries hard walls and
    costing §4's "broader object-level interaction than a shallow CNN". The
    configured depths must not silently regress to that."""
    for name, stage in (
        ("stage1", model_c.net.stage1),
        ("stage2", model_c.net.stage2),
        ("decoder", model_c.net.decoder),
    ):
        shifts = [b.shift_size for b in stage.blocks]
        assert any(s > 0 for s in shifts), f"{name} has no shifted block: {shifts}"
