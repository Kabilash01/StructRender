"""M10: TEMPORAL two-style B engines — one per style — with history fed back live.

Each engine is `renderer.temporal.live_step.LiveStep`: the confidence mask
(built in-graph with the training/scoring `confidence_mask`), the fusion and the
style-fixed decoder. Six inputs, two outputs (frame, this frame's features):

  x_t, x_prev (1,7,H,W) | feat_prev (1,C,h,w) | motion (1,2,H,W) UV
  no_prev (1,1,H,W) spawned-entity pixels | motion_valid (1,1,1,1)

Parity: a real game TEST pair through `srt_run` vs PyTorch for BOTH outputs —
TF32 off at 1e-3 (Decisions 48/52), TF32 on is what ships.
"""
from __future__ import annotations

import datetime as dt, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from inference.export.trt_compat import for_tensorrt
from renderer.temporal.live_step import LiveStep
from renderer.temporal.pair_dataset import TemporalFrameDataset
from renderer.training.dataset import target_dir_for
from renderer.training.train_baseline import ROOT, load_checkpoint
from scripts.m10_cpp_parity import ENGINES, SRT, build

NAMES_IN = ["x_t", "x_prev", "feat_prev", "motion", "no_prev", "motion_valid"]
NAMES_OUT = ["rgb", "feat"]


def real_inputs(m):
    sid = json.load(open(ROOT / "configs/data/game_split_v1.json"))["test_sessions"][1]
    s = StructuralSession(ROOT / "data_pipeline/recorder" / sid)
    ds = TemporalFrameDataset(s, target_dir_for(sid, "oil_impasto"), s.frame_ids())
    b = ds[ds.frame_ids.index(400)]
    assert b["has_history"]
    with torch.no_grad():
        fp, _ = m.encode(b["structure_prev"][None])
    return f"{sid} frames 399->400", [b["structure_curr"][None], b["structure_prev"][None], fp,
                                      b["motion_curr"][None], torch.zeros(1, 1, 288, 512), torch.ones(1, 1, 1, 1)]


def main(ckpt="checkpoints/game_b_two_style/best.pt") -> int:
    ENGINES.mkdir(parents=True, exist_ok=True)
    m = load_checkpoint(ROOT / ckpt, torch.device("cpu")).eval()
    what, ins = real_inputs(m)
    tmp = Path(tempfile.mkdtemp())
    files = []
    for n, t in zip(NAMES_IN, ins):
        (tmp / f"{n}.bin").write_bytes(t.float().numpy().tobytes()); files.append(str(tmp / f"{n}.bin"))
    rep = {"checkpoint": ckpt, "input": f"{what} (real game test pair)", "inputs": NAMES_IN,
           "outputs": NAMES_OUT, "feat_shape": list(ins[2].shape),
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "styles": {}}
    ok = True
    for idx, style in enumerate(m.style_ids):
        net = LiveStep(m, idx).eval()
        with torch.no_grad():
            ref = [r.numpy() for r in net(*ins)]
        onnx_p = tmp / f"live_{style}.onnx"
        torch.onnx.export(for_tensorrt(net), tuple(ins), str(onnx_p), input_names=NAMES_IN,
                          output_names=NAMES_OUT, opset_version=17, dynamo=False, do_constant_folding=True)
        rep["styles"][style] = {}
        for label, tf32 in (("fp32_strict", False), ("fp32_tf32", True)):
            eng = build(onnx_p, ENGINES / f"game_b2_live_{style}_{label}.engine", tf32)
            r = subprocess.run([str(SRT), str(eng), *files, "--out", str(tmp / "out.bin"), "--bench", "100"],
                               capture_output=True, text=True)
            diffs = {}
            if r.returncode == 0:
                for k, name in enumerate(NAMES_OUT):
                    o = np.frombuffer((tmp / ("out.bin" + (f".{k}" if k else ""))).read_bytes(),
                                      dtype=np.float32).reshape(ref[k].shape)
                    diffs[name] = float(np.abs(o - ref[k]).max())
            bench = [l for l in r.stdout.splitlines() if l.startswith("bench")]
            rep["styles"][style][label] = {"cpp_exit": r.returncode, "max_abs_diff_vs_torch": diffs,
                                           "engine": str(eng.relative_to(ROOT)), "bench": bench[0] if bench else None,
                                           "stderr": r.stderr[-300:]}
            print(f"[m10] {style:<22} {label:<12} exit {r.returncode}  {diffs}  {bench[0] if bench else ''}", flush=True)
        ok &= r.returncode == 0 and all(
            v < 1e-3 for v in rep["styles"][style]["fp32_strict"]["max_abs_diff_vs_torch"].values()) and \
            len(rep["styles"][style]["fp32_strict"]["max_abs_diff_vs_torch"]) == 2
    rep["strict_parity_pass"] = bool(ok)
    (ROOT / "eval/reports/m10_live_temporal_engines.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"[m10] strict parity (1e-3), both outputs, both styles: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
