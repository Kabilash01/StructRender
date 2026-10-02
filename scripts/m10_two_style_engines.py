"""M10 for two-style game B: one TensorRT engine PER STYLE, from the ONE checkpoint.

Each engine is the two-style model with its style index fixed (a 1-input /
1-output graph, exactly the contract the C++ runtime and GDExtension already
run), so a live style switch is the game choosing which engine renders the
next frame — no runtime change, both engines loaded once.

These engines run WITHOUT frame history (cold start every frame); the live
loop does not supply motion / confidence yet. The quality cost of that is
measured separately (eval/reports/game_b_two_style_test_*_nohistory.json).

Parity: a real game TEST frame through `srt_run` vs PyTorch, TF32 off at 1e-3
(Decisions 48/52, as m10_cpp_parity) and TF32 on (what ships).
"""
from __future__ import annotations

import datetime as dt, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from inference.export.export_onnx import export_onnx
from inference.export.trt_compat import for_tensorrt
from renderer.training.dataset import build_input
from renderer.training.train_baseline import ROOT, load_checkpoint
from scripts.m10_cpp_parity import ENGINES, SRT, build


class FixedStyle(torch.nn.Module):
    """structure -> rgb in one trained style; the style is a constant of the graph."""

    def __init__(self, model, style_idx: int):
        super().__init__()
        self.model, self.style_idx = model, int(style_idx)

    def forward(self, x):
        return self.model(x, style_idx=torch.full((x.shape[0],), self.style_idx, dtype=torch.long))


def main(ckpt="checkpoints/game_b_two_style/best.pt") -> int:
    ENGINES.mkdir(parents=True, exist_ok=True)
    m = load_checkpoint(ROOT / ckpt, torch.device("cpu")).eval()
    sid = json.load(open(ROOT / "configs/data/game_split_v1.json"))["test_sessions"][1]
    s = StructuralSession(ROOT / "data_pipeline/recorder" / sid)
    x = torch.from_numpy(build_input(s.load_structure(400)))[None].float()
    tmp = Path(tempfile.mkdtemp()); (tmp / "in.bin").write_bytes(x.numpy().astype(np.float32).tobytes())
    rep = {"checkpoint": ckpt, "input": f"{sid} frame 400 (real game test frame)", "history": "none (cold start)",
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "styles": {}}
    ok, shipped = True, {}
    for idx, style in enumerate(m.style_ids):
        net = FixedStyle(m, idx).eval()
        with torch.no_grad():
            ref = net(x).numpy()
        onnx_p = export_onnx(for_tensorrt(net), tmp / f"game_b_{style}.onnx")
        rep["styles"][style] = {}
        for label, tf32 in (("fp32_strict", False), ("fp32_tf32", True)):
            eng = build(onnx_p, ENGINES / f"game_b2_{style}_{label}.engine", tf32)
            r = subprocess.run([str(SRT), str(eng), str(tmp / "in.bin"), "--out", str(tmp / "out.bin")],
                               capture_output=True, text=True)
            out = np.frombuffer((tmp / "out.bin").read_bytes(), dtype=np.float32).reshape(ref.shape) \
                if r.returncode == 0 else None
            diff = float(np.abs(out - ref).max()) if out is not None else None
            rep["styles"][style][label] = {"cpp_exit": r.returncode, "max_abs_diff_vs_torch": diff,
                                           "engine": str(eng.relative_to(ROOT)),
                                           "engine_mb": round(eng.stat().st_size / 1e6, 1),
                                           "stdout": r.stdout[-300:], "stderr": r.stderr[-300:]}
            if label == "fp32_tf32" and out is not None:
                shipped[style] = out
            print(f"[m10] {style:<22} {label:<12} exit {r.returncode}  max|C++ - torch| {diff}", flush=True)
        ok &= (rep["styles"][style]["fp32_strict"]["max_abs_diff_vs_torch"] or 1) < 1e-3
    # The two styles must also differ through the C++ path, not only in PyTorch.
    if len(shipped) == 2:
        a, b = shipped.values()
        rep["cpp_style_difference_mean_abs"] = float(np.abs(a - b).mean())
        print(f"[m10] C++ oil vs watercolour mean |diff| {rep['cpp_style_difference_mean_abs']:.4f}")
    rep["strict_parity_pass"] = bool(ok)
    (ROOT / "eval/reports/m10_two_style_engines.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"[m10] strict parity (1e-3), both styles: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
