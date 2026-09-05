"""SD 1.5 + ControlNet-depth img2img teacher (KB Decision 31).

The control signal is the dataset's **own exported depth buffer**, not an
estimated depth map — structural preservation is a conditioning input, not a
hope. The buffer is converted to the inverse-depth ("near is bright") image
ControlNet-depth was trained on; background (`+inf`) becomes 0.

Reproducibility: generation is seed-pinned and reproducible on a fixed
GPU/driver, but is **not** bit-exact across machines. That is exactly why
targets are cached and version-keyed rather than regenerated on demand
(KB §3C.5). ``BIT_EXACT`` is therefore False and the contract tests only assert
bit-exactness for the ``deterministic`` backend.

Licences (KB Decision 31): SD 1.5 CreativeML OpenRAIL-M, ControlNet Apache-2.0.
Both are recorded in the target manifest provenance.
"""

from __future__ import annotations

import numpy as np

from ..adapter import FrameStructure, TeacherBackend, TeacherOutput
from ..styles import Style


def depth_to_control_image(depth: np.ndarray, *, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Exported metric depth (``+inf`` == background) -> (H, W, 3) uint8
    inverse-depth control image, bright = near."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    finite = np.isfinite(d) & (d > 0.0)
    disp = np.zeros_like(d)
    disp[finite] = 1.0 / d[finite]
    if finite.any():
        lo = float(np.percentile(disp[finite], lo_pct))
        hi = float(np.percentile(disp[finite], hi_pct))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        disp = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)
        disp[~finite] = 0.0
    out = np.rint(disp * 255.0).astype(np.uint8)
    return np.repeat(out[..., None], 3, axis=-1)


class SD15ControlNetDepthBackend(TeacherBackend):
    NAME = "sd15_controlnet_depth"
    BACKEND_VERSION = "sd15-controlnet-depth-1"
    BIT_EXACT = False
    # one generator seeded per STYLE, not per frame (see generate() below), so
    # identical inputs give identical output within a run
    CONTENT_ADDRESSABLE = True

    def __init__(self, style: Style):
        super().__init__(style)
        self._pipe = None
        self._torch = None
        self._resolved: dict = {}

    # -- model loading (lazy: --dry-run never reaches here) ------------------
    def _load(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetImg2ImgPipeline,
                UniPCMultistepScheduler,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the sd15_controlnet_depth teacher needs torch + diffusers "
                "(pip install torch diffusers transformers accelerate safetensors); "
                "use the 'deterministic' backend for tests/CI"
            ) from exc

        t = self.style.teacher
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            t["controlnet_model_id"], revision=t["controlnet_revision"], torch_dtype=dtype
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            t["model_id"],
            revision=t["revision"],
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to(device)
        if device == "cuda":
            # 8 GB laptop card: cheap insurance, no effect on output values.
            # Guarded by hasattr — these helpers have moved around across
            # diffusers releases and a missing one must not fail a run.
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

        self._torch = torch
        self._pipe = pipe
        self._resolved = {
            "device": device,
            "torch_dtype": str(dtype).replace("torch.", ""),
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        }
        return pipe

    # -- adapter contract ----------------------------------------------------
    def generate(
        self,
        native_rgb: np.ndarray,
        structure: FrameStructure,
        style: Style,
    ) -> TeacherOutput:
        from PIL import Image

        pipe = self._load()
        torch = self._torch
        p = style.params

        H, W = native_rgb.shape[0], native_rgb.shape[1]
        init = Image.fromarray(native_rgb, mode="RGB")
        control = Image.fromarray(depth_to_control_image(structure.depth), mode="RGB")

        # One generator seed for the whole style (not per frame): img2img from a
        # fixed seed keeps consecutive frames from flickering independently.
        gen = torch.Generator(device=pipe.device.type).manual_seed(int(style.seed))

        result = pipe(
            prompt=style.prompt,
            negative_prompt=style.negative_prompt or None,
            image=init,
            control_image=control,
            height=H,
            width=W,
            strength=float(p["strength"]),
            num_inference_steps=int(p["num_inference_steps"]),
            guidance_scale=float(p["guidance_scale"]),
            controlnet_conditioning_scale=float(p["controlnet_conditioning_scale"]),
            generator=gen,
            output_type="pil",
        )
        out = np.asarray(result.images[0].convert("RGB"), dtype=np.uint8)
        self._check_output(out, native_rgb)

        t = style.teacher
        return TeacherOutput(
            styled_rgb=out,
            metadata={
                "backend": self.NAME,
                "backend_version": self.BACKEND_VERSION,
                "bit_exact": False,
                "research_grade": True,
                "model_id": t["model_id"],
                "revision": t["revision"],
                "controlnet_model_id": t["controlnet_model_id"],
                "controlnet_revision": t["controlnet_revision"],
                "licences": {
                    "sd15": "CreativeML OpenRAIL-M",
                    "controlnet": "Apache-2.0",
                },
                "control": "exported_depth_buffer_inverse_normalised",
                "params": {
                    "num_inference_steps": int(p["num_inference_steps"]),
                    "guidance_scale": float(p["guidance_scale"]),
                    "strength": float(p["strength"]),
                    "controlnet_conditioning_scale": float(p["controlnet_conditioning_scale"]),
                },
                **self._resolved,
            },
        )
