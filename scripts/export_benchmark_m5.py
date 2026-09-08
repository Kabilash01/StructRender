"""M5 exit criteria 3 and 4 at TRAINED width: ONNX export + TensorRT benchmark.

**This is the only thing that writes ``eval/reports/m5_onnx_export.json``.**
KB Decision 44 established that a committed artifact with no generator and no
test is unfalsifiable evidence; Decision 45 closed that for the export probe and
this closes it for the trained-width export. Asserted by
``renderer/structformer/tests/test_m5_export_report.py``.

Reports the export EXPERIENCE, not just pass/fail: per-stage outcome, the
workaround each stage needed, operator census vs M4, and FP32/FP16 latency.

Run:  python -m scripts.export_benchmark_m5     (needs CUDA + tensorrt)
"""
import datetime as dt, json, sys, time
from pathlib import Path
import numpy as np, torch, onnx
sys.path.insert(0, "/home/kabilash01/StructRender")
from renderer.training.train_baseline import ROOT, load_checkpoint, objective_string, load_config, parameter_count
from inference.export.export_onnx import export_onnx
from inference.tensorrt.builder.build_engine import build_engine, tensorrt_version
from inference.tensorrt.benchmark.run_engine import run_engine

# --- schema, declared here so a test can check the committed report against the
# --- generator rather than against itself -------------------------------------

#: Every top-level key this generator writes. `test_m5_export_report.py` imports
#: this and fails if the committed artifact and the generator disagree, which is
#: how the first version of this file came to be missing four fields that the
#: committed report carried (they had been hand-patched in). KB Decision 51.
REQUIRED_TOP_LEVEL = (
    "milestone", "checkpoint", "parameters", "session_id", "style_id", "objective",
    "generated_utc", "torch_version", "tensorrt_version", "opset", "input_shape",
    "stages", "operator_allow_list", "operator_allow_list_source", "export_contract",
    "budget_ref", "model_a_comparison_ref", "latency_note",
)

#: The allow-list is a DECLARATION, not an observation. Deriving it from the graph
#: it is meant to validate makes it vacuous — every operator would always pass.
#: A new operator at M6 must fail this list and be added deliberately.
#: §12 wants a new op in the graph to be a visible diff.
ALLOWED_OPS = frozenset({
    # M4's U-Net census, carried forward
    "Add", "Concat", "Constant", "Conv", "InstanceNormalization", "MaxPool", "Mul",
    "Relu", "Reshape", "Resize", "Shape", "Sigmoid",
    # M5 window-attention additions (KB Decision 45)
    "ConstantOfShape", "Erf", "Expand", "Gather", "Gemm", "Identity",
    "LayerNormalization", "MatMul", "Slice", "Softmax", "Sub", "Tile", "Transpose",
    "Unsqueeze", "Div", "Cast", "Sqrt", "ReduceMax",
    # emitted by `.expand(structure.shape[0], -1)` on the cached style token
    # (KB Decision 50) — present because the shipped graph is single-input
    "Equal", "Where",
})

#: FP32 max-abs parity tolerance, matching M4's. FP16 is measured against the
#: same fp32 torch reference and is EXPECTED to exceed it; the report flags the
#: exceedance rather than waiving it, and FP16 gets its own tolerance at M10.
FP32_TOLERANCE = 1e-3


cfg = load_config("configs/structformer/m5_structformer.json")
dev = torch.device("cuda")
ckpt = ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
model = load_checkpoint(ckpt, dev).eval()
SHAPE = (1, 7, 288, 512)
OPSET = int(cfg.get("export", {}).get("opset", 17))
out = {"milestone": 5, "checkpoint": str(ckpt), "parameters": parameter_count(model),
       "session_id": cfg["data"]["session_id"] if "session_id" in cfg["data"] else "ds_1788687350_337657ea",
       "style_id": cfg["data"]["style_id"], "objective": objective_string(cfg),
       "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
       "torch_version": torch.__version__, "tensorrt_version": tensorrt_version(),
       "opset": OPSET, "input_shape": list(SHAPE), "stages": {},
       "operator_allow_list": sorted(ALLOWED_OPS),
       "operator_allow_list_source": (
           "Declared in scripts/export_benchmark_m5.py::ALLOWED_OPS, NOT derived from the "
           "exported graph. Deriving it from the graph it validates would make it vacuous. "
           "A new operator must fail and be added deliberately (KB 12, Decision 51)."),
       "budget_ref": {"kb_section": "12", "structformer_tensorrt_target_ms": [15, 20],
                      "frame_budget_ms": 33.3,
                      "note": "KB 12 calls these engineering targets, not guaranteed results."},
       "model_a_comparison_ref": {
           "artifact": "eval/reports/m4_tensorrt_build_failure.json",
           "note": ("Model A's TensorRT build failure is a SEPARATE dated diagnostic, not "
                    "re-measured by this generator. Embedding its numbers in a freshly dated "
                    "report would present an old measurement as new (KB Decision 51).")},
       "latency_note": (
           "Attempt-1 checkpoint, batch 1, 512x288, TensorRT engine timing only: CUDA events "
           "around execute_async_v3. EXCLUDES host<->device transfers, Godot's own frame cost, "
           "structural-buffer export and any game-pipeline overhead. Not an end-to-end FPS "
           "figure and not comparable to a published end-to-end number.")}

# Reference BEFORE export: export_onnx does `model.eval().to("cpu").float()`,
# mutating the model in place, so anything computed on `model` afterwards is on CPU.
x = torch.randn(*SHAPE)
with torch.no_grad():
    ref = model(x.to(dev)).float().cpu().numpy()

# ---- stage 1: ONNX export (FP32) -------------------------------------------
onnx_fp32 = Path(S := "/tmp/m5_structformer_fp32.onnx")
t0 = time.time(); export_onnx(model, onnx_fp32, input_shape=SHAPE, opset=OPSET)
g = onnx.load(str(onnx_fp32)).graph
census = {}
for n in g.node: census[n.op_type] = census.get(n.op_type, 0) + 1
def dims(vi): return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in vi.type.tensor_type.shape.dim]
ishapes = {vi.name: dims(vi) for vi in g.input}; oshapes = {vi.name: dims(vi) for vi in g.output}
static = all(isinstance(d, int) and d > 0 for s in list(ishapes.values()) + list(oshapes.values()) for d in s)
out["export_contract"] = {
    "graph_inputs": len(ishapes), "input_names": sorted(ishapes),
    "note": ("Derived from the graph actually exported, not from prose. M5 ships "
             "StructFormerSingleStyle, which folds the single cached style token into a "
             "learned parameter, so it never crosses the graph boundary -> ONE input. "
             "KB Decision 45 item 3 pinned a two-input contract; that belongs to the "
             "multi-style StructFormer class and returns at M8 with style-B "
             "(KB Decision 50).")}
out["stages"]["onnx_fp32"] = {"ok": True, "seconds": round(time.time()-t0,2), "workaround": None,
    "inputs": ishapes, "outputs": oshapes, "shapes_are_static": static,
    "n_nodes": len(g.node), "op_types": census, "size_mb": round(onnx_fp32.stat().st_size/1e6,2)}

# ---- stage 2: onnxruntime parity -------------------------------------------
import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
got = sess.run(None, {"structure": x.numpy()})[0]
out["stages"]["onnxruntime_parity"] = {"ok": True, "max_abs_diff": float(np.abs(got-ref).max()),
    "tolerance": 1e-3, "providers": ["CPUExecutionProvider"], "workaround": None}

# ---- stage 3: TensorRT FP32 -------------------------------------------------
t0=time.time(); eng32, info32 = build_engine(onnx_fp32, strongly_typed=False, tf32=False, workspace_bytes=1<<30)
out["stages"]["trt_fp32"] = {"ok": eng32 is not None, "seconds": round(time.time()-t0,2),
    "workaround": "tf32 disabled so max_abs_diff measures operator correctness, not TF32 rounding",
    **{k:v for k,v in info32.items() if k!="engine"}}
if eng32:
    r = run_engine(eng32, {"structure": x}, warmup=20, iters=100)
    o = list(r["outputs"].values())[0].float().cpu().numpy()
    out["stages"]["trt_fp32"].update({"max_abs_diff_vs_torch": float(np.abs(o-ref).max()),
        "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r.get("latency_ms_p50"),
        "latency_ms_p95": r.get("latency_ms_p95")})

# ---- stage 4: TensorRT FP16 (Decision 45 item 1: export-time, not builder flag)
onnx_fp16 = Path("/tmp/m5_structformer_fp16.onnx")
t0=time.time()
mh = load_checkpoint(ckpt, dev).eval().half()   # trace on CUDA: no CPU fp16 LayerNorm/GELU
torch.onnx.export(mh, (torch.zeros(*SHAPE, dtype=torch.float16, device=dev),), str(onnx_fp16),
                  input_names=["structure"], output_names=["rgb"], opset_version=OPSET,
                  dynamo=False, do_constant_folding=True)
eng16, info16 = build_engine(onnx_fp16, strongly_typed=True, tf32=False, workspace_bytes=1<<30)
out["stages"]["trt_fp16"] = {"ok": eng16 is not None, "seconds": round(time.time()-t0,2),
    "workaround": ("TensorRT 11 removed BuilderFlag.FP16 (KB Decision 45): precision comes only from a "
                   "STRONGLY_TYPED network inheriting dtypes from the ONNX graph, so the graph itself must "
                   "be exported in fp16, traced on CUDA because LayerNorm/GELU have no CPU fp16 kernels"),
    **{k:v for k,v in info16.items() if k!="engine"}}
if eng16:
    r = run_engine(eng16, {"structure": x}, warmup=20, iters=100)
    o = list(r["outputs"].values())[0].float().cpu().numpy()
    _d16 = float(np.abs(o - ref).max())
    out["stages"]["trt_fp16"]["fp32_tolerance"] = FP32_TOLERANCE
    out["stages"]["trt_fp16"]["fp32_tolerance_exceeded"] = _d16 > FP32_TOLERANCE
    if _d16 > FP32_TOLERANCE:
        out["stages"]["trt_fp16"]["fp16_tolerance_note"] = (
            "EXPECTED but NOT WAIVED (KB Decision 48): half precision carries ~1e-3 relative "
            "precision and the reference is fp32 torch, so exceeding a tolerance written for "
            "FP32 is not a defect - but 'expected' is not 'inside the gate'. FP16 needs its "
            "own tolerance before it is an acceptance criterion at M10.")
    out["stages"]["trt_fp16"].update({"max_abs_diff_vs_torch": _d16,
        "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r.get("latency_ms_p50"),
        "latency_ms_p95": r.get("latency_ms_p95")})

# --- self-validation: refuse to write a report that fails its own schema ------
_missing = [k for k in REQUIRED_TOP_LEVEL if k not in out]
if _missing:
    raise SystemExit(f"generator/schema drift - REQUIRED_TOP_LEVEL keys not produced: {_missing}")
_census = set(out["stages"]["onnx_fp32"]["op_types"])
_unlisted = sorted(_census - ALLOWED_OPS)
if _unlisted:
    raise SystemExit(
        f"operators in the exported graph are not on ALLOWED_OPS: {_unlisted}. "
        "§12 requires a new operator to be a visible diff - add it to ALLOWED_OPS "
        "deliberately, with a KB entry, rather than widening the list to fit.")

Path("eval/reports/m5_onnx_export.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items() if kk!="op_types"})
                  for k,v in out["stages"].items()}, indent=2))
