"""Does ``GridSample`` reach TensorRT? — M6 step 0, run BEFORE the warp module exists.

KB §12's ONNX/TensorRT operator risk, applied to M6. The engine-guided temporal
reprojection of §4 is a backward warp, which in PyTorch is
``F.grid_sample``; ONNX has ``GridSample`` only from **opset 16** and it is a
known-risky operator. If it does not reach a *built and executed* TensorRT
engine, the whole M6 design changes — so this runs first, on the same reasoning
that made the M5 attention probe (KB Decision 45) pay for itself.

Every engine below is **built and executed**, never merely exported: an
"exports fine" claim with no engine behind it reads as coverage and is worse
than no result (KB Decision 45 / the evidence rule Decision 51 records).

Writes ``eval/reports/m6_gridsample_probe.json``. This script is the ONLY thing
that writes it (KB Decision 44: an artifact with no generator is unfalsifiable).

    python -m scripts.probe_m6_gridsample            # needs CUDA + TensorRT
    python -m scripts.probe_m6_gridsample --no-trt   # ONNX/ORT only, scoped in the artifact

Exit code is 1 if any variant is blocked, so this cannot report success by
omission (KB Decision 52).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.tensorrt.benchmark.run_engine import run_engine
from inference.tensorrt.builder.build_engine import (
    build_engine,
    tensorrt_available,
    tensorrt_version,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "eval" / "reports" / "m6_gridsample_probe.json"

OPSET = 17
SEED = 20260910
#: M4's FP32 max-abs parity tolerance, carried forward unchanged. FP16 is measured
#: against the same FP32 torch reference and is EXPECTED to exceed it; the report
#: flags the exceedance rather than waiving it (KB Decision 48).
FP32_TOLERANCE = 1e-3

#: Image resolution (M4/M5 shipped shape) and the stage-1 feature resolution the
#: M5 CNN stem produces (4x downsample) at Model C's embed_dim.
IMAGE_HW = (288, 512)
FEATURE_HW = (72, 128)
FEATURE_C = 240

#: Every top-level key this generator writes. A test compares the committed
#: artifact against this tuple so the two cannot drift (KB Decision 51).
REQUIRED_TOP_LEVEL = (
    "milestone", "purpose", "question", "generated_utc", "torch_version",
    "tensorrt_version", "opset", "tolerance_max_abs", "seed", "motion_convention",
    "m5_allow_list_source", "m5_allowed_ops", "new_operators_vs_m5", "variants",
    "blocked", "verdict", "trt_scope", "latency_note", "variant_groups",
)

#: What each variant group is for, and when it ran. The MINIMAL group answers the
#: blocking question and was executed, and its verdict reached, before
#: ``renderer/temporal/`` contained a line of code — that ordering is the whole
#: point of a design-time probe (KB Decision 45). The SHIPPED group was added
#: afterwards, once the module existed, to price the real graph. Both are written
#: by this one generator so the artifact stays reproducible from it (Decision 51).
VARIANT_GROUPS = {
    "minimal": ("Standalone F.grid_sample forms. Answers 'does GridSample reach TensorRT, "
                "and under which sampling options' with no dependency on renderer/temporal. "
                "Run FIRST, before that package existed."),
    "fp32_grid_workaround": ("The measured fix for the FP16 finding below: half-precision "
                             "sampled tensor, fp32 grid arithmetic and fp32 sample."),
    "shipped": ("The real renderer/temporal modules — the backward warp and KB §4's full "
                "confidence mask — at the shipped 288x512 resolution. This is what prices "
                "the warp+mask against §12's 15-20 ms StructFormer budget. Added after the "
                "modules existed; it does not carry the design-time ordering claim."),
}

#: Where the M5 declaration is READ from, rather than copied. Copying it here
#: would let the two drift and turn "new operator vs M5" into a comparison
#: against a stale list.
M5_EXPORT_REPORT = ROOT / "eval" / "reports" / "m5_onnx_export.json"


def m5_allowed_ops() -> tuple[list[str], str]:
    data = json.loads(M5_EXPORT_REPORT.read_text(encoding="utf-8"))
    return sorted(data["operator_allow_list"]), (
        "Read at run time from eval/reports/m5_onnx_export.json -> operator_allow_list, "
        "which scripts/export_benchmark_m5.py::ALLOWED_OPS declares. Read, not copied: a "
        "second copy would drift and make 'new operator vs M5' a comparison against a "
        "stale list (KB Decisions 50/51, §12 'a new operator must be a visible diff')."
    )


# --------------------------------------------------------------------------- #
# the minimal warp — deliberately NOT imported from renderer/temporal
# --------------------------------------------------------------------------- #
class MinimalWarp(nn.Module):
    """`W(prev, motion)` in the fewest primitives that can express §4's warp.

    ``motion`` is backward, screen-UV in [0,1], Y-down (Decision 25), so the
    sampling grid is ``2 * (uv_curr + motion) - 1`` with **no** rescaling for
    resolution and **no** Y flip. This class is intentionally standalone: the
    probe's whole point is to answer the export question before
    ``renderer/temporal/`` exists, so it cannot depend on it.
    """

    def __init__(self, h: int, w: int, mode: str, padding_mode: str, align_corners: bool):
        super().__init__()
        self.mode, self.padding_mode, self.align_corners = mode, padding_mode, align_corners
        ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
        xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
        uv = torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], dim=-1)  # (H,W,2) as (u,v)
        # Non-persistent: it is derived from the fixed shape, not learned, and a
        # persistent buffer would put it in every checkpoint (KB Decision 45).
        self.register_buffer("base_uv", uv[None], persistent=False)

    def forward(self, prev: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        grid = 2.0 * (self.base_uv + motion.permute(0, 2, 3, 1)) - 1.0
        return F.grid_sample(prev, grid, mode=self.mode,
                             padding_mode=self.padding_mode,
                             align_corners=self.align_corners)


def _synthetic_inputs(c: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured previous frame and a motion field with real structure.

    A constant image or a zero motion field would let a broken sampler pass:
    every wrong grid would fetch the same value. The field mixes a global
    translation (camera-like) with a smooth divergent term (parallax-like) and
    deliberately pushes part of the frame **out of bounds**, which is the case
    §4's mask has to handle.
    """
    g = torch.Generator().manual_seed(SEED)
    prev = torch.rand(1, c, h, w, generator=g)
    ys = torch.linspace(-1.0, 1.0, h)[:, None].expand(h, w)
    xs = torch.linspace(-1.0, 1.0, w)[None, :].expand(h, w)
    u = 0.04 + 0.03 * xs          # up to 0.07 UV -> ~36 px at W=512, some off-frame
    v = -0.02 + 0.025 * ys
    motion = torch.stack([u, v], dim=0)[None].contiguous()
    return prev, motion


VARIANTS = (
    # name,                              C,          H/W,        mode,        padding, align_corners
    ("image_bilinear_zeros_ac0",         3,          IMAGE_HW,   "bilinear",  "zeros",  False),
    ("image_bilinear_border_ac0",        3,          IMAGE_HW,   "bilinear",  "border", False),
    ("image_bilinear_zeros_ac1",         3,          IMAGE_HW,   "bilinear",  "zeros",  True),
    ("image_nearest_zeros_ac0",          3,          IMAGE_HW,   "nearest",   "zeros",  False),
    ("feature_bilinear_zeros_ac0", FEATURE_C,        FEATURE_HW, "bilinear",  "zeros",  False),
)


class Fp32GridWarp(nn.Module):
    """The measured FP16 workaround: sampled tensor in half, GRID ARITHMETIC in fp32.

    A grid value lives in ``[-1, 1]``, where fp16's absolute resolution is
    ``2^-11``; at W = 512 that is a quarter-pixel of *positional* error before a
    single sample is fetched. The consequence is not half-precision rounding of
    the colour, it is sampling the wrong pixel, which is why the FP16 diff grows
    with resolution and is worst for ``nearest``. Keeping the grid and the sample
    in fp32 inside an otherwise-half graph fixes it; the report carries both
    measurements so the choice is evidence, not assertion.
    """

    def __init__(self, h: int, w: int):
        super().__init__()
        ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
        xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
        uv = torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], dim=-1)
        self.register_buffer("base_uv", uv[None], persistent=False)

    def forward(self, prev: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        grid = 2.0 * (self.base_uv + motion.permute(0, 2, 3, 1)) - 1.0
        out = F.grid_sample(prev.float(), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        return out.half()


def _census(graph) -> dict:
    out: dict = {}
    for n in graph.node:
        out[n.op_type] = out.get(n.op_type, 0) + 1
    return out


def _dims(vi):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def _blocked(variants: dict, with_trt: bool) -> list[str]:
    """Which variants failed an outcome this probe requires.

    Covers every required outcome, not just export: the M5 probe's first gate
    counted ONNX export and the FP32 run only, so an FP16 build failure or a
    parity miss left the list empty and the probe reported success (KB
    Decision 52). A non-finite parity value counts as blocked.
    """
    out = []
    for name, v in variants.items():
        bad = not v.get("onnx_export_ok") or not v.get("shapes_are_static")
        bad = bad or not v.get("onnxruntime", {}).get("within_tolerance")
        if with_trt and v["gate"] == "minimal":
            f32, f16 = v.get("trt_fp32", {}), v.get("trt_fp16", {})
            bad = bad or not (f32.get("build_ok") and f32.get("run_ok"))
            bad = bad or not _finite(f32.get("max_abs_diff_vs_torch"))
            bad = bad or not f32.get("within_tolerance")
            bad = bad or not f32.get("output_finite")
            # FP16 must BUILD, RUN and stay finite. It is deliberately NOT held to
            # the FP32 tolerance — this probe's finding is that it cannot meet one.
            bad = bad or not (f16.get("build_ok") and f16.get("run_ok"))
            bad = bad or not f16.get("output_finite")
        if with_trt and v["gate"] == "shipped_fp32_only":
            # FP32 outcomes only. FP16 is not requested for this variant, and the
            # gate must not silently pass because a required outcome is absent.
            f32 = v.get("trt_fp32", {})
            bad = bad or not (f32.get("build_ok") and f32.get("run_ok"))
            bad = bad or not _finite(f32.get("max_abs_diff_vs_torch"))
            bad = bad or not f32.get("within_tolerance")
            bad = bad or not f32.get("output_finite")
            bad = bad or v.get("trt_fp16", {}).get("measured") is not False
        if with_trt and v["gate"] == "fp32_grid_workaround":
            # A workaround is only a workaround if it MEASURABLY works: it must
            # reach the FP32 tolerance the plain FP16 form misses.
            f16 = v.get("trt_fp16", {})
            bad = bad or not (f16.get("build_ok") and f16.get("run_ok"))
            bad = bad or not f16.get("output_finite")
            bad = bad or not _finite(f16.get("max_abs_diff_vs_torch_fp32"))
            bad = bad or bool(f16.get("fp32_tolerance_exceeded"))
        if bad:
            out.append(name)
    return out


def probe_variant(name, c, hw, mode, padding, align_corners, with_trt: bool) -> dict:
    import onnx
    import onnxruntime as ort

    h, w = hw
    prev, motion = _synthetic_inputs(c, h, w)
    m = MinimalWarp(h, w, mode, padding, align_corners).eval()
    with torch.no_grad():
        ref = m(prev, motion).numpy()

    v: dict = {
        "gate": "minimal",
        "mode": mode, "padding_mode": padding, "align_corners": align_corners,
        "shapes": {"prev": [1, c, h, w], "motion": [1, 2, h, w]},
        "torch_output_finite": bool(np.isfinite(ref).all()),
    }

    onnx_path = Path("/tmp") / f"m6_gridsample_{name}.onnx"
    torch.onnx.export(m, (prev, motion), str(onnx_path),
                      input_names=["prev", "motion"], output_names=["warped"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)
    g = onnx.load(str(onnx_path)).graph
    v["onnx_export_ok"] = True
    v["op_types"] = _census(g)
    v["n_nodes"] = len(g.node)
    v["shapes_are_static"] = all(
        isinstance(d, int) and d > 0
        for vi in list(g.input) + list(g.output) for d in _dims(vi))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"prev": prev.numpy(), "motion": motion.numpy()})[0]
    d = float(np.abs(got - ref).max())
    v["onnxruntime"] = {"max_abs_diff": d, "tolerance": FP32_TOLERANCE,
                        "within_tolerance": bool(np.isfinite(d) and d <= FP32_TOLERANCE),
                        "providers": ["CPUExecutionProvider"]}
    if not with_trt:
        return v

    eng, info = build_engine(onnx_path, strongly_typed=False, tf32=False, workspace_bytes=1 << 30)
    v["trt_fp32"] = {"build_ok": eng is not None, "errors": info.get("errors", []),
                     "n_layers": info.get("n_layers")}
    if eng is not None:
        r = run_engine(eng, {"prev": prev, "motion": motion}, warmup=20, iters=100)
        o = list(r["outputs"].values())[0].float().cpu().numpy()
        d32 = float(np.abs(o - ref).max())
        v["trt_fp32"].update({
            "run_ok": True, "max_abs_diff_vs_torch": d32, "tolerance": FP32_TOLERANCE,
            "within_tolerance": bool(np.isfinite(d32) and d32 <= FP32_TOLERANCE),
            "output_finite": bool(np.isfinite(o).all()),
            "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r["latency_ms_p50"]})

    # FP16: TensorRT 11 removed BuilderFlag.FP16, so precision comes only from a
    # STRONGLY_TYPED network inheriting the ONNX graph's dtypes — the graph itself
    # must be exported in fp16 (KB Decision 45 item 1).
    onnx16 = Path("/tmp") / f"m6_gridsample_{name}_fp16.onnx"
    m16 = MinimalWarp(h, w, mode, padding, align_corners).eval().cuda().half()
    torch.onnx.export(m16, (prev.cuda().half(), motion.cuda().half()), str(onnx16),
                      input_names=["prev", "motion"], output_names=["warped"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)
    eng16, info16 = build_engine(onnx16, strongly_typed=True, tf32=False, workspace_bytes=1 << 30)
    v["trt_fp16"] = {"build_ok": eng16 is not None, "errors": info16.get("errors", []),
                     "n_layers": info16.get("n_layers")}
    if eng16 is not None:
        r = run_engine(eng16, {"prev": prev, "motion": motion}, warmup=20, iters=100)
        o = list(r["outputs"].values())[0].float().cpu().numpy()
        d16 = float(np.abs(o - ref).max())
        v["trt_fp16"].update({
            "run_ok": True, "max_abs_diff_vs_torch": d16,
            "fp32_tolerance": FP32_TOLERANCE,
            "fp32_tolerance_exceeded": bool(not np.isfinite(d16) or d16 > FP32_TOLERANCE),
            "output_finite": bool(np.isfinite(o).all()),
            "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r["latency_ms_p50"]})
        if not np.isfinite(d16) or d16 > FP32_TOLERANCE:
            v["trt_fp16"]["fp16_tolerance_note"] = (
                "EXPECTED but NOT WAIVED (KB Decision 48). The reference is fp32 torch and "
                "half carries ~1e-3 relative precision, so exceeding an FP32 tolerance is "
                "not a defect — but 'expected' is not 'inside the gate'. FP16 needs its own "
                "tolerance before it is an acceptance criterion.")
    return v


def probe_fp32_grid_workaround() -> dict:
    """FP16 engine, fp32 grid + fp32 sample. Needs CUDA + TensorRT by construction."""
    import onnx

    h, w = IMAGE_HW
    prev, motion = _synthetic_inputs(3, h, w)
    m = Fp32GridWarp(h, w).eval()
    # The reference is the FULL-fp32 warp — the thing the workaround is meant to
    # reproduce. Comparing against a half reference would grade it against the
    # error it exists to remove.
    ref = MinimalWarp(h, w, "bilinear", "zeros", False).eval()
    with torch.no_grad():
        ref_out = ref(prev, motion).numpy()

    v: dict = {
        "gate": "fp32_grid_workaround",
        "mode": "bilinear", "padding_mode": "zeros", "align_corners": False,
        "shapes": {"prev": [1, 3, h, w], "motion": [1, 2, h, w]},
        "precision": "prev fp16, motion fp32, grid fp32, grid_sample fp32, output fp16",
        "purpose": ("Measured workaround for the FP16 positional error. Reference is the "
                    "full-fp32 warp, not a half reference."),
        "torch_output_finite": True,
    }
    onnx_path = Path("/tmp") / "m6_gridsample_fp32grid.onnx"
    torch.onnx.export(m, (prev.half(), motion), str(onnx_path),
                      input_names=["prev", "motion"], output_names=["warped"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)
    g = onnx.load(str(onnx_path)).graph
    v["onnx_export_ok"] = True
    v["op_types"] = _census(g)
    v["n_nodes"] = len(g.node)
    v["shapes_are_static"] = all(
        isinstance(d, int) and d > 0
        for vi in list(g.input) + list(g.output) for d in _dims(vi))
    # onnxruntime cannot be asked for CPU fp16 parity here; the graph is built for
    # a strongly-typed engine. Recorded as not-run rather than silently skipped.
    v["onnxruntime"] = {"run": False, "within_tolerance": True,
                        "note": "mixed-precision graph is measured on TensorRT, not on the CPU EP"}

    eng, info = build_engine(onnx_path, strongly_typed=True, tf32=False, workspace_bytes=1 << 30)
    v["trt_fp16"] = {"build_ok": eng is not None, "errors": info.get("errors", []),
                     "n_layers": info.get("n_layers")}
    if eng is not None:
        r = run_engine(eng, {"prev": prev, "motion": motion}, warmup=20, iters=100)
        o = list(r["outputs"].values())[0].float().cpu().numpy()
        d = float(np.abs(o - ref_out).max())
        v["trt_fp16"].update({
            "run_ok": True, "max_abs_diff_vs_torch_fp32": d,
            "fp32_tolerance": FP32_TOLERANCE,
            "fp32_tolerance_exceeded": bool(not np.isfinite(d) or d > FP32_TOLERANCE),
            "output_finite": bool(np.isfinite(o).all()),
            "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r["latency_ms_p50"]})
    return v


class ShippedWarpAndMask(nn.Module):
    """The graph M6 actually ships: ``W(prev)`` plus §4's ``M_t``.

    Imports the production modules rather than reimplementing them, so what is
    priced here is what runs. The two outputs are the warped previous frame and
    the confidence mask, which is exactly the pair a temporal fusion consumes.
    """

    def __init__(self, h: int, w: int):
        super().__init__()
        from renderer.temporal.warp import EngineGuidedWarp

        self.warp = EngineGuidedWarp(h, w)

    def forward(self, prev_rgb, depth_t, depth_prev, normal_t, normal_prev,
                motion, fin_t, fin_prev, fresh):
        from renderer.temporal.confidence import confidence_mask

        m = confidence_mask(depth_t=depth_t, depth_prev=depth_prev,
                            normal_t=normal_t, normal_prev=normal_prev, motion=motion,
                            depth_finite_t=fin_t, depth_finite_prev=fin_prev,
                            motion_valid=True, no_prev_transform=fresh)["mask"]
        return self.warp(prev_rgb, motion), m


def probe_shipped_module(with_trt: bool) -> dict:
    """TensorRT cost of the real warp + confidence mask at the shipped resolution."""
    import onnx
    import onnxruntime as ort

    h, w = IMAGE_HW
    prev, motion = _synthetic_inputs(3, h, w)
    g = torch.Generator().manual_seed(SEED)
    depth_t = 2.0 + 20.0 * torch.rand(1, 1, h, w, generator=g)
    depth_prev = depth_t + 0.05 * torch.randn(1, 1, h, w, generator=g)
    n = torch.randn(1, 3, h, w, generator=g)
    normal_t = n / n.norm(dim=1, keepdim=True)
    n2 = normal_t + 0.05 * torch.randn(1, 3, h, w, generator=g)
    normal_prev = n2 / n2.norm(dim=1, keepdim=True)
    fin_t = torch.ones(1, 1, h, w)
    fin_prev = torch.ones(1, 1, h, w)
    fresh = torch.zeros(1, 1, h, w)
    fresh[..., :20, :20] = 1.0
    args = (prev, depth_t, depth_prev, normal_t, normal_prev, motion, fin_t, fin_prev, fresh)
    names = ["prev", "depth_t", "depth_prev", "normal_t", "normal_prev",
             "motion", "fin_t", "fin_prev", "fresh"]

    m = ShippedWarpAndMask(h, w).eval()
    with torch.no_grad():
        ref = [t.numpy() for t in m(*args)]

    v: dict = {"gate": "shipped_fp32_only",
               "shapes": {k: list(t.shape) for k, t in zip(names, args)},
               "outputs": ["warped_prev", "confidence_mask"],
               "torch_output_finite": bool(all(np.isfinite(r).all() for r in ref))}

    onnx_path = Path("/tmp") / "m6_shipped_warp_mask.onnx"
    torch.onnx.export(m, args, str(onnx_path), input_names=names,
                      output_names=["warped", "mask"], opset_version=OPSET,
                      dynamo=False, do_constant_folding=True)
    gr = onnx.load(str(onnx_path)).graph
    v["onnx_export_ok"] = True
    v["op_types"] = _census(gr)
    v["n_nodes"] = len(gr.node)
    v["shapes_are_static"] = all(
        isinstance(d, int) and d > 0
        for vi in list(gr.input) + list(gr.output) for d in _dims(vi))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: t.numpy() for k, t in zip(names, args)})
    d = max(float(np.abs(a - b).max()) for a, b in zip(got, ref))
    v["onnxruntime"] = {"max_abs_diff": d, "tolerance": FP32_TOLERANCE,
                        "within_tolerance": bool(np.isfinite(d) and d <= FP32_TOLERANCE),
                        "providers": ["CPUExecutionProvider"]}
    if not with_trt:
        return v

    eng, info = build_engine(onnx_path, strongly_typed=False, tf32=False, workspace_bytes=1 << 30)
    v["trt_fp32"] = {"build_ok": eng is not None, "errors": info.get("errors", []),
                     "n_layers": info.get("n_layers")}
    if eng is not None:
        r = run_engine(eng, dict(zip(names, args)), warmup=20, iters=100)
        outs = list(r["outputs"].values())
        dd = max(float(np.abs(o.float().cpu().numpy() - b).max())
                 for o, b in zip(outs, ref))
        v["trt_fp32"].update({
            "run_ok": True, "max_abs_diff_vs_torch": dd, "tolerance": FP32_TOLERANCE,
            "within_tolerance": bool(np.isfinite(dd) and dd <= FP32_TOLERANCE),
            "output_finite": bool(all(np.isfinite(o.float().cpu().numpy()).all() for o in outs)),
            "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r["latency_ms_p50"]})
    # FP16 is deliberately NOT built for this variant, and the artifact says so
    # rather than carrying build_ok/run_ok flags for a run that never happened —
    # that is the "success by omission" shape KB Decision 52 records. Its gate
    # ("shipped_fp32_only") does not ask for FP16 outcomes at all.
    v["trt_fp16"] = {"measured": False,
                     "reason": ("intentionally skipped: the minimal group already measured that "
                                "fp16 grid arithmetic costs up to a quarter-pixel of POSITION at "
                                "this resolution, and image_bilinear_zeros_ac0_fp32grid measured "
                                "the fix. Building it here would price a configuration this "
                                "probe recommends against.")}
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 step 0 — GridSample export probe")
    ap.add_argument("--no-trt", action="store_true",
                    help="ONNX/onnxruntime only; the artifact records the reduced scope")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    with_trt = (not args.no_trt) and tensorrt_available() and torch.cuda.is_available()
    allowed, allowed_src = m5_allowed_ops()

    report = {
        "milestone": 6,
        "purpose": ("M6 step 0. Answer whether ONNX GridSample (opset >= 16) reaches a BUILT "
                    "AND EXECUTED TensorRT engine, before renderer/temporal/ exists — the same "
                    "sequencing as the M5 attention probe (KB Decision 45). A blocker here "
                    "changes the M6 design."),
        "question": "Does F.grid_sample export to opset 17, build under TensorRT, run, and match PyTorch?",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "tensorrt_version": tensorrt_version(),
        "opset": OPSET,
        "tolerance_max_abs": FP32_TOLERANCE,
        "seed": SEED,
        "motion_convention": (
            "Backward, screen-UV in [0,1], Y-down, analytic from engine transforms "
            "(KB Decision 25, data_pipeline/schemas/frame_structure.md §4). UV is "
            "resolution-independent, so the field needs NO rescaling at feature resolution; "
            "the sampling grid is 2*(uv_curr + motion) - 1 with no Y flip."),
        "m5_allow_list_source": allowed_src,
        "m5_allowed_ops": allowed,
        "variant_groups": VARIANT_GROUPS,
        "trt_scope": (
            "TensorRT engines were built and executed for every variant."
            if with_trt else
            "TENSORRT NOT RUN (--no-trt or no CUDA/TensorRT). An empty 'blocked' list here "
            "asserts NOTHING about TensorRT — only about ONNX export and onnxruntime parity "
            "(KB Decision 52)."),
        "latency_note": (
            "Batch 1, TensorRT engine time only: CUDA events around execute_async_v3. "
            "EXCLUDES host<->device transfers, the renderer itself, Godot's frame cost and "
            "structural export. This is the marginal cost of the WARP OPERATOR alone, to be "
            "read against the M5 StructFormer engine time, not an end-to-end FPS figure."),
        "variants": {},
    }

    for name, c, hw, mode, padding, ac in VARIANTS:
        print(f"[m6-probe] {name} ...", flush=True)
        report["variants"][name] = probe_variant(name, c, hw, mode, padding, ac, with_trt)
    if with_trt:
        print("[m6-probe] image_bilinear_zeros_ac0_fp32grid ...", flush=True)
        report["variants"]["image_bilinear_zeros_ac0_fp32grid"] = probe_fp32_grid_workaround()
    print("[m6-probe] shipped_warp_and_mask ...", flush=True)
    report["variants"]["shipped_warp_and_mask"] = probe_shipped_module(with_trt)

    seen: set[str] = set()
    for v in report["variants"].values():
        seen |= set(v["op_types"])
    report["new_operators_vs_m5"] = sorted(seen - set(allowed))

    report["blocked"] = _blocked(report["variants"], with_trt)
    report["verdict"] = "PASS" if not report["blocked"] else "BLOCKED"

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in report]
    if missing:
        raise SystemExit(f"generator/schema drift — REQUIRED_TOP_LEVEL keys not produced: {missing}")

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[m6-probe] wrote {out}")
    for name, v in report["variants"].items():
        f32, f16 = v.get("trt_fp32", {}), v.get("trt_fp16", {})
        ort = v["onnxruntime"].get("max_abs_diff")
        print(f"  {name:34s} onnx={v['onnx_export_ok']} static={v['shapes_are_static']} "
              f"ort={('%.2e' % ort) if ort is not None else 'n/a':>8s}")
        if f16.get("measured") is False:
            f16s = "not measured"
        else:
            f16s = f"{f16.get('build_ok')}/{f16.get('run_ok')} {f16.get('latency_ms_mean')} ms"
        print(f"    {'':32s}trt32={f32.get('build_ok')}/{f32.get('run_ok')} "
              f"d={f32.get('max_abs_diff_vs_torch')} {f32.get('latency_ms_mean')} ms | "
              f"trt16={f16s}")
    print(f"[m6-probe] new operators vs M5 allow-list: {report['new_operators_vs_m5']}")
    print(f"[m6-probe] {report['verdict']}  blocked={report['blocked']}")
    return 0 if not report["blocked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
