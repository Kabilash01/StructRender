"""M10: the StructRenderer GDExtension, run inside Godot, vs PyTorch on a real frame.

Checks the in-game path end to end — Godot -> GDExtension -> C++ input builder
(a re-implementation of `build_input`) -> TensorRT -> RGB8 — against PyTorch
rendering the same recorded frame. Tolerance: 1 RGB8 level (quantisation), on the
strict-FP32 engine.
"""
import json, os, subprocess, tempfile
from pathlib import Path
import numpy as np, torch
from data_pipeline.teacher.structural import StructuralSession
from renderer.training.dataset import build_input
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, load_checkpoint

sp = load_session_split(ROOT / "configs/data/session_split_v2.json")
s = StructuralSession(ROOT / "data_pipeline/recorder" / sp["test_sessions"][0])
st = s.load_structure(400)
tmp = Path(tempfile.mkdtemp())
dep = st.depth[..., 0] if st.depth.ndim == 3 else st.depth
(tmp / "depth.bin").write_bytes(dep.astype(np.float32).tobytes())
(tmp / "normal.bin").write_bytes(st.normal.astype(np.float32).tobytes())
(tmp / "albedo.bin").write_bytes(st.albedo.astype(np.float32).tobytes())
m = load_checkpoint(ROOT / "checkpoints/ladder_v3_a/best.pt", torch.device("cpu")).eval()
with torch.no_grad():
    ref = m(torch.from_numpy(build_input(st))[None]).numpy()[0]
ref8 = np.clip(np.rint(np.transpose(ref, (1, 2, 0)) * 255), 0, 255).astype(np.int16)
eng = ROOT / "artifacts/engines/model_a_fp32_strict.engine"
env = dict(os.environ, SR_PARITY_DIR=str(tmp), SR_ENGINE=str(eng))
r = subprocess.run(["godot", "--headless", "--path", str(ROOT / "game_project"),
                    "--script", "res://tests/runtime/gdext_parity.gd"], env=env, capture_output=True, text=True, timeout=300)
line = [l for l in (r.stdout + r.stderr).splitlines() if "PARITY" in l]
print("godot exit", r.returncode, "|", line[-1] if line else (r.stdout + r.stderr)[-400:])
if r.returncode == 0:
    out = np.frombuffer((tmp / "rgb8.bin").read_bytes(), dtype=np.uint8).reshape(288, 512, 3).astype(np.int16)
    d = np.abs(out - ref8)
    rep = {"frame": f"{s.session_id}/400", "engine": eng.name, "max_rgb8_diff": int(d.max()),
           "pixels_diff_gt1": int((d.max(-1) > 1).sum()), "godot_line": line[-1] if line else None}
    (ROOT / "eval/reports/m10_gdext_parity.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"max |Godot - torch| = {rep['max_rgb8_diff']} RGB8 levels; pixels off by >1: {rep['pixels_diff_gt1']}"
          f" -> {'PASS' if rep['max_rgb8_diff'] <= 1 else 'FAIL'}")
