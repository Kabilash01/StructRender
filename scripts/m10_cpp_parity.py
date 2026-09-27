"""M10: build a Model's TensorRT engine and prove the C++ runtime reproduces PyTorch.

Exports through `for_tensorrt` (Decision 84), builds the engine with the same
TensorRT the C++ runtime links, feeds a REAL test frame to `srt_run`, and
compares its output to PyTorch. Two builds: TF32 off (the correctness check,
tolerance 1e-3 as in Decisions 48/52) and TF32 on (what actually ships).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from inference.export.export_onnx import export_onnx
from inference.export.trt_compat import for_tensorrt
from renderer.training.dataset import build_input
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, load_checkpoint

SRT = ROOT / "inference/cpp_runtime/build/srt_run"
ENGINES = ROOT / "artifacts/engines"


def build(onnx_path, out, tf32):
    import tensorrt as trt
    lg = trt.Logger(trt.Logger.ERROR); b = trt.Builder(lg); n = b.create_network(0)
    assert trt.OnnxParser(n, lg).parse(Path(onnx_path).read_bytes())
    c = b.create_builder_config(); c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if not tf32:
        c.clear_flag(trt.BuilderFlag.TF32)
    s = b.build_serialized_network(n, c)
    assert s is not None, "engine build failed"
    out.write_bytes(bytes(s)); return out


def main(tag="a", ckpt="checkpoints/ladder_v3_a/best.pt") -> int:
    ENGINES.mkdir(parents=True, exist_ok=True)
    m = load_checkpoint(ROOT / ckpt, torch.device("cpu")).eval()
    mt = for_tensorrt(m)
    tmp = Path(tempfile.mkdtemp())
    onnx_p = export_onnx(mt, tmp / f"model_{tag}.onnx")
    sp = load_session_split(ROOT / "configs/data/session_split_v2.json")
    s = StructuralSession(ROOT / "data_pipeline/recorder" / sp["test_sessions"][0])
    x = torch.from_numpy(build_input(s.load_structure(400)))[None].float()
    with torch.no_grad():
        ref = m(x).numpy()
    (tmp / "in.bin").write_bytes(x.numpy().astype(np.float32).tobytes())
    rep = {"model": tag, "checkpoint": ckpt, "input": f"{s.session_id} frame 400 (real test frame)",
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "builds": {}}
    for label, tf32 in (("fp32_strict", False), ("fp32_tf32", True)):
        eng = build(onnx_p, ENGINES / f"model_{tag}_{label}.engine", tf32)
        r = subprocess.run([str(SRT), str(eng), str(tmp / "in.bin"), "--out", str(tmp / "out.bin")],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        out = np.frombuffer((tmp / "out.bin").read_bytes(), dtype=np.float32).reshape(ref.shape) if ok else None
        diff = float(np.abs(out - ref).max()) if ok else None
        rep["builds"][label] = {"cpp_exit": r.returncode, "max_abs_diff_vs_torch": diff,
                                "finite": bool(np.isfinite(out).all()) if ok else None,
                                "engine_mb": round(eng.stat().st_size / 1e6, 1),
                                "stderr": r.stderr[-300:]}
        print(f"[m10] {tag} {label:<12} C++ exit {r.returncode}  max|C++ - torch| "
              f"{diff if diff is None else f'{diff:.2e}'}  finite {rep['builds'][label]['finite']}", flush=True)
    rep["strict_parity_pass"] = (rep["builds"]["fp32_strict"]["max_abs_diff_vs_torch"] or 1) < 1e-3
    (ROOT / f"eval/reports/m10_cpp_parity_{tag}.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"[m10] strict parity (1e-3): {'PASS' if rep['strict_parity_pass'] else 'FAIL'}")
    return 0 if rep["strict_parity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
