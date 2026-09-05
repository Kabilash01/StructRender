"""Dependency-free ``deterministic`` teacher backend (KB Decision 31).

Classical edge-preserving smoothing + posterisation + ink outlines over the
native RGB. Its whole job is to make the adapter contract, the cache, and the
alignment-validation code paths testable on a clean checkout with no model
download, and to be **bit-exact** across runs so those tests are reproducible.

It is NOT a research-grade teacher. Targets used to report results must come
from the diffusion backend; ``scripts/test_teacher.sh`` and the generator both
say so, and the target manifest records ``research_grade: false``.
"""

from __future__ import annotations

import numpy as np

from ..adapter import FrameStructure, TeacherBackend, TeacherOutput
from ..imageops import bilateral3_step, gradient_magnitude, luma
from ..styles import Style


class DeterministicBackend(TeacherBackend):
    NAME = "deterministic"
    BACKEND_VERSION = "deterministic-1"
    BIT_EXACT = True
    # no per-frame randomness at all
    CONTENT_ADDRESSABLE = True

    def generate(
        self,
        native_rgb: np.ndarray,
        structure: FrameStructure,
        style: Style,
    ) -> TeacherOutput:
        p = style.params
        levels = int(p["levels"])
        smooth_iters = int(p["smooth_iters"])
        sigma_color = float(p["sigma_color"])
        edge_strength = float(p["edge_strength"])
        edge_threshold = float(p["edge_threshold"])
        if levels < 2:
            raise ValueError(f"params.levels must be >= 2, got {levels}")

        src = native_rgb.astype(np.float64) / 255.0
        img = src
        for _ in range(smooth_iters):
            img = bilateral3_step(img, sigma_color)

        # posterise onto `levels` evenly spaced values in [0, 1]
        q = np.clip(np.floor(img * levels), 0, levels - 1) / (levels - 1)

        # Ink outlines are traced from the ORIGINAL luma, not the smoothed one:
        # smoothing is what a cel-shader is allowed to destroy, edge position is
        # not. (The seed scene is near-monochrome white-box geometry — its
        # boundaries are 2-5/255 luma steps, so tracing the smoothed image would
        # erase most of the layout.)
        mag, _gx, _gy = gradient_magnitude(luma(src))
        # ramp from `edge_threshold` to 5x it, so the ink is scaled to the
        # scene's own contrast rather than to the (never-reached) [0,1] range
        ink = np.clip((mag - edge_threshold) / max(1e-6, 4.0 * edge_threshold), 0.0, 1.0)
        shade = np.clip(1.0 - edge_strength * ink, 0.0, 1.0)

        out = np.clip(np.rint(q * shade[..., None] * 255.0), 0, 255).astype(np.uint8)
        self._check_output(out, native_rgb)

        return TeacherOutput(
            styled_rgb=out,
            metadata={
                "backend": self.NAME,
                "backend_version": self.BACKEND_VERSION,
                "bit_exact": True,
                "research_grade": False,
                "device": "cpu",
                "params": {
                    "levels": levels,
                    "smooth_iters": smooth_iters,
                    "sigma_color": sigma_color,
                    "edge_strength": edge_strength,
                    "edge_threshold": edge_threshold,
                },
            },
        )
