"""Close the loop: the C++ LIVE temporal frame == the scored PyTorch model.

Reads two consecutive frames dumped by game_project/tests/runtime/
live_temporal_parity.gd (structure + GDScript motion from export_frame, and
the C++ render_temporal() output of frame f1, whose history was f0 inside C++).
Rebuilds f1 in PyTorch with LiveStep on the same inputs and compares uint8 RGB.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
from renderer.temporal.live_step import LiveStep
from renderer.training.dataset import build_input
from renderer.training.train_baseline import ROOT, load_checkpoint

H, W = 288, 512

def frame(d: Path, tag: str):
    f = lambda n, c: np.frombuffer((d / f"{tag}_{n}.bin").read_bytes(), dtype=np.float32).reshape(H, W, c) if c > 1 else \
        np.frombuffer((d / f"{tag}_{n}.bin").read_bytes(), dtype=np.float32).reshape(H, W)
    class S: pass
    s = S(); s.depth, s.normal, s.albedo_or_material = f("depth", 1), f("normal", 3), f("albedo", 3)
    s.albedo = s.albedo_or_material
    x = torch.from_numpy(build_input(s))[None].float()
    mo = torch.from_numpy(np.ascontiguousarray(f("motion", 2).transpose(2, 0, 1)))[None]
    return x, mo

def main(dump="eval/reports/m10_live_temporal_dump") -> int:
    d = ROOT / dump
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m = load_checkpoint(ROOT / "checkpoints/game_b_two_style/best.pt", torch.device("cpu")).eval()
    x0, _ = frame(d, "f0"); x1, mo1 = frame(d, "f1")
    style = int((d / "f1_style.txt").read_text())
    with torch.no_grad():
        f0, _ = m.encode(x0)
        rgb, _ = LiveStep(m, style)(x1, x0, f0, mo1, torch.zeros(1, 1, H, W), torch.ones(1, 1, 1, 1))
    ref = np.rint(rgb[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.int16)
    cpp = np.frombuffer((d / "f1_rgb.bin").read_bytes(), dtype=np.uint8).reshape(H, W, 3).astype(np.int16)
    diff = np.abs(cpp - ref)
    rep = {"style": style, "max_abs_diff_uint8": int(diff.max()), "pixels_diff_gt1": int((diff.max(-1) > 1).sum()),
           "mean_abs_diff_uint8": float(diff.mean())}
    print(json.dumps(rep))
    (ROOT / "eval/reports/m10_live_temporal_parity.json").write_text(json.dumps(rep, indent=2) + "\n")
    return 0 if rep["max_abs_diff_uint8"] <= 1 else 1

if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
