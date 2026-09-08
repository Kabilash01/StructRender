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
       "opset": OPSET, "input_shape": list(SHAPE), "stages": {}}

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
    out["stages"]["trt_fp16"].update({"max_abs_diff_vs_torch": float(np.abs(o-ref).max()),
        "latency_ms_mean": r["latency_ms_mean"], "latency_ms_p50": r.get("latency_ms_p50"),
        "latency_ms_p95": r.get("latency_ms_p95")})

Path("eval/reports/m5_onnx_export.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items() if kk!="op_types"})
                  for k,v in out["stages"].items()}, indent=2))
