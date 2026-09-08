"""The real teacher (SD 1.5 + ControlNet-depth).

The depth→control-image conversion is pure numpy and is always tested.

The generation test is the one part of the M3 suite that **can** touch the GPU.
It runs only when torch/diffusers, the pinned weights, *and* enough free VRAM
are all present; otherwise it SKIPS with a reason — never silently passes (same
discipline as ``scripts/test_teacher.sh`` exit 2). So the suite is GPU-free on a
clean checkout and GPU-bound only on a machine that has already fetched the
weights. That asymmetry is deliberate but it must never make the verdict depend
on what else is using the card — hence the VRAM guard and the OOM→skip below.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.teacher.adapter import make_backend
from data_pipeline.teacher.backends.sd15_controlnet import depth_to_control_image
from data_pipeline.teacher.structural import StructuralSession
from data_pipeline.teacher.styles import load_style

from .conftest import CONFIG_DIR


# --------------------------------------------------------------------------- #
# depth conditioning (no torch needed)
# --------------------------------------------------------------------------- #
def test_control_image_is_inverse_depth_with_black_background():
    """KB Decision 31: the control signal is the dataset's OWN exported depth
    buffer, not an estimated depth map. Near must be bright, `+inf` background
    must be black, and the output must be a 3-channel uint8 image."""
    depth = np.array([[1.0, 2.0, np.inf], [4.0, 8.0, np.inf]], dtype=np.float32)
    ctrl = depth_to_control_image(depth)

    assert ctrl.shape == (2, 3, 3)
    assert ctrl.dtype == np.uint8
    assert (ctrl[..., 0] == ctrl[..., 1]).all() and (ctrl[..., 1] == ctrl[..., 2]).all()
    assert ctrl[0, 2, 0] == 0 and ctrl[1, 2, 0] == 0, "+inf background must be black"
    assert ctrl[0, 0, 0] > ctrl[0, 1, 0] > ctrl[1, 0, 0], "nearer must be brighter"


def test_control_image_uses_the_real_depth_buffer(mini_session):
    session = StructuralSession(mini_session)
    structure = session.load_structure(0)
    ctrl = depth_to_control_image(structure.depth)

    assert ctrl.shape == (structure.depth.shape[0], structure.depth.shape[1], 3)
    bg = ~np.isfinite(structure.depth)
    assert bg.any(), "the seed scene has sky"
    assert (ctrl[bg] == 0).all()
    assert ctrl[~bg].max() > 0


def test_control_image_survives_an_all_background_frame():
    ctrl = depth_to_control_image(np.full((4, 4), np.inf, dtype=np.float32))
    assert (ctrl == 0).all()


# --------------------------------------------------------------------------- #
# real generation (skipped when torch/diffusers or the weights are absent)
# --------------------------------------------------------------------------- #
# Measured headroom for the pinned pipeline at 512x288 fp16 on the 8 GB target
# card: ~3.5 GiB resident during the 990-frame production run. Anything under
# this and we are sharing the card with another job (a concurrent generate, a
# compositor, another test session).
_MIN_FREE_VRAM_BYTES = 4.0 * 1024**3


def _teacher_available(style) -> str | None:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError:
        return "torch/diffusers not installed"

    # VRAM is an environment condition, not a property of the code under test.
    # Without this check the suite's verdict depends on whatever else happens to
    # be on the GPU: the 990-frame production run holding 3.5 GiB turned this
    # test red with an OOM that said nothing about the backend. Skipping with a
    # reason keeps the suite deterministic and never reports a false PASS
    # (same discipline as scripts/test_teacher.sh exit 2).
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if free < _MIN_FREE_VRAM_BYTES:
            return (f"only {free / 1024**3:.2f} GiB VRAM free of {total / 1024**3:.2f} GiB — "
                    f"need {_MIN_FREE_VRAM_BYTES / 1024**3:.1f} GiB; another job is using the GPU")
    # Only a probe: diffusers downloads a subset of each repo, so a full
    # snapshot check would false-negative. One pinned config file per repo is
    # enough to tell "the weights were fetched at this revision" from "nothing
    # is cached".
    try:
        from huggingface_hub import try_to_load_from_cache

        probes = (
            (style.teacher["model_id"], "model_index.json", style.teacher["revision"]),
            (style.teacher["controlnet_model_id"], "config.json", style.teacher["controlnet_revision"]),
        )
        for repo, filename, rev in probes:
            if not isinstance(try_to_load_from_cache(repo, filename, revision=rev), str):
                return f"pinned teacher weights not in the local HF cache ({repo}@{rev[:8]})"
    except Exception as exc:
        return f"cannot probe the HF cache: {exc}"
    return None


def test_diffusion_teacher_produces_an_aligned_frame(mini_session):
    style = load_style("oil_impasto", CONFIG_DIR)
    reason = _teacher_available(style)
    if reason:
        pytest.skip(reason)

    session = StructuralSession(mini_session)
    backend = make_backend(style)
    native = session.load_native_rgb(0)
    try:
        out = backend.generate(native, session.load_structure(0), style)
    except Exception as exc:  # noqa: BLE001 — narrowed immediately below
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            pytest.skip(f"CUDA OOM during generation — the GPU is shared: {exc}")
        raise

    assert out.styled_rgb.dtype == np.uint8
    assert out.styled_rgb.shape == native.shape
    assert out.metadata["research_grade"] is True
    assert out.metadata["bit_exact"] is False
    assert out.metadata["revision"] == style.teacher["revision"]
    assert out.metadata["control"] == "exported_depth_buffer_inverse_normalised"
    # it must actually restyle, not pass the native frame through
    assert not np.array_equal(out.styled_rgb, native)
