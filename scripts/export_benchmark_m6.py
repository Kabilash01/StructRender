"""M6 export + TensorRT benchmark for Model D — the milestone's missing evidence.

**This is the only thing that writes ``eval/reports/m6_onnx_export.json``**
(KB Decisions 51/58: a committed artifact needs a generator and a test).

Why it exists: §10 does not list export among M6's deliverables, but §12 requires
the export path proven, M10 depends on it, and every other ladder rung has it.
Model D shipped with its config marked ``NOT YET EXERCISED``.

Model D also introduces a runtime requirement no earlier rung had: its forward is
**four-input**, and the caller must carry ``prev_features (1, 480, 36, 64)``
*between frames*. The M5 runtime had no cross-frame state at all. That contract is
recorded here so M10 designs for it rather than discovering it.

    python -m scripts.export_benchmark_m6      (needs CUDA + tensorrt)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

from inference.tensorrt.benchmark.run_engine import run_engine
from inference.tensorrt.builder.build_engine import build_engine, tensorrt_available, tensorrt_version
from renderer.training.train_baseline import ROOT, load_checkpoint, load_config

SHAPE = (1, 7, 288, 512)
OPSET = 17
FP32_TOLERANCE = 1e-3
SEED = 20260910

#: Declared, never derived from the graph it validates (KB Decision 51). M6 adds
#: GridSample and the mask's boolean ops on top of M5's census.
ALLOWED_OPS = frozenset({
    "Add", "Cast", "Concat", "Constant", "ConstantOfShape", "Conv", "Div", "Equal",
    "Erf", "Expand", "Gather", "Gemm", "Greater", "Identity", "InstanceNormalization",
    "IsNaN", "LayerNormalization", "MatMul", "MaxPool", "Mul", "Not", "Or",
    "ReduceMax", "ReduceSum", "Relu", "Reshape", "Resize", "Shape", "Sigmoid",
    "Slice", "Softmax", "Sqrt", "Sub", "Tile", "Transpose", "Unsqueeze", "Where",
    "GridSample",
    # M6 addition, added deliberately after the allow-list guard rejected it on the
    # first run (§12: a new operator must be a VISIBLE diff). AveragePool is the
    # mask's area-pooling to feature resolution — Decision 55 chose area over
    # bilinear because a feature cell covers 8x8 pixels and bilinear would report
    # full confidence for a mostly-disoccluded cell.
    "AveragePool",
})

REQUIRED_TOP_LEVEL = (
    "milestone", "checkpoint", "session_id", "style_id", "generated_utc",
    "torch_version", "tensorrt_version", "opset", "stages", "operator_allow_list",
    "runtime_contract", "latency_note", "m5_comparison",
)


class _Encoder(nn.Module):
    """Frame t's history producer: run once per frame, output cached for t+1."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, structure):
        feats, _skip = self.d.encode(structure)
        return feats


class _Renderer(nn.Module):
    """The four-input graph the runtime actually calls each frame."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, structure, prev_features, motion, confidence):
        return self.d(structure, prev_features, motion, confidence)


def _graph_facts(path: Path) -> dict:
    g = onnx.load(str(path)).graph
    census: dict = {}
    for n in g.node:
        census[n.op_type] = census.get(n.op_type, 0) + 1

    def dims(vi):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in vi.type.tensor_type.shape.dim]

    ins = {vi.name: dims(vi) for vi in g.input}
    outs = {vi.name: dims(vi) for vi in g.output}
    return {
        "inputs": ins, "outputs": outs, "n_nodes": len(g.node), "op_types": census,
        "shapes_are_static": all(isinstance(d, int) and d > 0
                                 for s in list(ins.values()) + list(outs.values()) for d in s),
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }


def _stage(name, module, args, input_names, out: dict, refs: dict) -> None:
    onnx_path = Path(f"/tmp/m6_{name}.onnx")
    torch.onnx.export(module, args, str(onnx_path), input_names=input_names,
                      output_names=["out"], opset_version=OPSET, dynamo=False,
                      do_constant_folding=True)
    facts = _graph_facts(onnx_path)
    with torch.no_grad():
        ref = module(*args).float().cpu().numpy()
    refs[name] = ref
    stage = {"onnx_export_ok": True, **facts}

    eng, info = build_engine(onnx_path, strongly_typed=False, tf32=False, workspace_bytes=1 << 30)
    stage["trt_fp32"] = {"build_ok": eng is not None, "errors": info.get("errors", []),
                         "n_layers": info.get("n_layers")}
    if eng is not None:
        feeds = dict(zip(input_names, [a.cpu() for a in args]))
        r = run_engine(eng, feeds, warmup=20, iters=100)
        got = list(r["outputs"].values())[0].float().cpu().numpy()
        d = float(np.abs(got - ref).max())
        stage["trt_fp32"].update({
            "run_ok": True, "max_abs_diff_vs_torch": d, "tolerance": FP32_TOLERANCE,
            "within_tolerance": bool(np.isfinite(d) and d <= FP32_TOLERANCE),
            "output_finite": bool(np.isfinite(got).all()),
            "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r.get("latency_ms_p50"),
        })
    out["stages"][name] = stage


def main() -> int:
    cfg = load_config(ROOT / "configs/temporal/m6_temporal.json")
    out_path = ROOT / "eval" / "reports" / "m6_onnx_export.json"
    ck = ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
    out = {
        "milestone": 6, "checkpoint": str(ck.relative_to(ROOT)),
        "session_id": cfg["data"].get("session_id", "ds_1788687350_337657ea"),
        "style_id": cfg["data"]["style_id"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__, "tensorrt_version": tensorrt_version(),
        "opset": OPSET, "operator_allow_list": sorted(ALLOWED_OPS),
        "operator_allow_list_source": (
            "Declared in scripts/export_benchmark_m6.py::ALLOWED_OPS, NOT derived from the graph "
            "it validates (KB Decision 51). GridSample and the mask's boolean ops are M6's "
            "additions over M5's census."),
        "runtime_contract": {
            "graphs": 2,
            "encoder": "structure(1,7,288,512) -> prev_features(1,480,36,64); run ONCE per frame",
            "renderer": ("structure(1,7,288,512), prev_features(1,480,36,64), motion(1,2,288,512), "
                         "confidence(1,1,288,512) -> rgb(1,3,288,512)"),
            "cross_frame_state": "prev_features (1,480,36,64) float — 1.11 MiB at fp32, 0.55 at fp16",
            "note": ("M10 CONSEQUENCE, recorded because no earlier rung needed it: the runtime must "
                     "hold one feature tensor from the previous frame. The M5 runtime was stateless. "
                     "Cold start (frame 0, or motion_valid false) passes no history and reduces "
                     "Model D exactly to Model C."),
        },
        "latency_note": (
            "Batch 1, 288x512, TensorRT engine time only — CUDA events around execute_async_v3. "
            "EXCLUDES host<->device transfers, the structural export, Godot's frame cost and the "
            "cross-frame copy. Not an end-to-end FPS figure. The per-frame runtime cost is "
            "encoder + renderer, since the encoder runs once per frame to produce the NEXT frame's "
            "history."),
        "stages": {},
    }
    if not (tensorrt_available() and torch.cuda.is_available()):
        out["skipped"] = "tensorrt or CUDA unavailable"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        return 0

    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    d = load_checkpoint(ck, dev).eval()
    structure = torch.randn(*SHAPE, device=dev)
    with torch.no_grad():
        feats, _ = d.encode(structure)
    motion = torch.randn(1, 2, 288, 512, device=dev) * 0.01
    conf = torch.rand(1, 1, 288, 512, device=dev)

    refs: dict = {}
    _stage("encoder", _Encoder(d).eval(), (structure,), ["structure"], out, refs)
    _stage("renderer", _Renderer(d).eval(),
           (structure, feats, motion, conf),
           ["structure", "prev_features", "motion", "confidence"], out, refs)

    enc = out["stages"]["encoder"].get("trt_fp32", {}).get("latency_ms_mean")
    ren = out["stages"]["renderer"].get("trt_fp32", {}).get("latency_ms_mean")
    if enc and ren:
        out["per_frame_ms_fp32"] = enc + ren
        out["m5_comparison"] = {
            "m5_structformer_fp32_ms": 13.48, "m5_structformer_fp16_ms": 4.22,
            "m6_per_frame_fp32_ms": enc + ren,
            "note": ("M5's figure is a single one-input graph; M6's is encoder + renderer, because "
                     "the history must be produced each frame. Compare FP32 to FP32 only — M6 has "
                     "no FP16 engine here."),
        }
    else:
        out["m5_comparison"] = {"note": "not computed — an engine did not build or run"}

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in out]
    if missing:
        raise SystemExit(f"generator/schema drift — missing {missing}")
    for name, st in out["stages"].items():
        unlisted = sorted(set(st["op_types"]) - ALLOWED_OPS)
        if unlisted:
            raise SystemExit(f"{name}: operators not on ALLOWED_OPS: {unlisted} — §12 requires a "
                             "new operator to be a visible diff, added deliberately")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[m6-export] wrote {out_path}")
    for name, st in out["stages"].items():
        t = st.get("trt_fp32", {})
        print(f"  {name:9s} onnx={st['onnx_export_ok']} static={st['shapes_are_static']} "
              f"trt={t.get('build_ok')} maxdiff={t.get('max_abs_diff_vs_torch')} "
              f"ms={t.get('latency_ms_mean')}")
    if "per_frame_ms_fp32" in out:
        print(f"  per-frame FP32 total: {out['per_frame_ms_fp32']:.2f} ms  (M5 FP32 13.48 ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
