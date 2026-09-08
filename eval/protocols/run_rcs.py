"""Compute RCS for the native frames, the teacher targets, and each ladder model.

KB §11's headline figure: RCS versus time outside view. Writes
``eval/reports/rcs.json``. Model rows need a checkpoint and (for speed) CUDA;
native and target rows need neither.

    python -m eval.protocols.run_rcs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.metrics.rcs import AppearanceEncoder, evaluate_source, summarise
from renderer.training.dataset import build_input
from renderer.training.train_baseline import ROOT, load_checkpoint, load_config, resolve_data

MODELS = {
    "m4_unet_baseline": "configs/training/m4_unet_baseline.json",
    "m5_structformer_attempt1": "configs/structformer/m5_structformer.json",
    "m5_structformer_attempt2_warmup": "configs/structformer/m5_structformer_warmup.json",
}


def _target_loader(target_dir: Path, session):
    index = {int(r["frame_id"]): r for r in
             (json.loads(l) for l in (target_dir / "manifest.jsonl").read_text().splitlines() if l.strip())}

    def load(fid: int) -> np.ndarray:
        rel = index[fid]["target_rgb_path"]
        return np.asarray(Image.open(target_dir / rel).convert("RGB"), dtype=np.float32) / 255.0
    return load


def _model_loader(model, session, device):
    def load(fid: int) -> np.ndarray:
        x = torch.from_numpy(build_input(session.load_structure(fid)))[None].to(device)
        with torch.no_grad():
            y = model(x)[0].float().clamp(0, 1).cpu().numpy()
        return np.transpose(y, (1, 2, 0))
    return load


def main() -> int:
    ap = argparse.ArgumentParser(description="RCS vs time outside view (KB §11)")
    ap.add_argument("--config", default=MODELS["m4_unet_baseline"])
    ap.add_argument("--out", default="eval/reports/rcs.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session, target_dir, _ = resolve_data(cfg)
    enc = AppearanceEncoder(device)

    report = {
        "metric": "RCS_i(dt) = cos(E(I_i^t), E(I_i^{t+dt}))  — KB §11",
        "session_id": session.session_id,
        "style_id": cfg["data"]["style_id"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "encoder": "AlexNet(IMAGENET1K_V1) conv taps, masked-average-pooled, per-layer L2, concatenated",
        "encoder_note": ("E is a masked deep-feature descriptor, not a crop: only the entity's own "
                         "pixels contribute, and the mask always comes from the ENGINE's entity map "
                         "so it cannot move with a render's own errors (KB Decision 53)."),
        "sources": {},
    }

    print("[rcs] native ...", flush=True)
    report["sources"]["native_engine"] = summarise(
        evaluate_source(session, session.load_native_rgb, enc))
    print("[rcs] teacher target ...", flush=True)
    report["sources"]["teacher_target"] = summarise(
        evaluate_source(session, _target_loader(target_dir, session), enc))

    for name, cfg_path in MODELS.items():
        mcfg = load_config(cfg_path)
        ck = ROOT / mcfg["train"]["checkpoint_dir"] / "best.pt"
        if not ck.is_file():
            print(f"[rcs] {name}: no checkpoint at {ck} — skipped", flush=True)
            continue
        print(f"[rcs] {name} ...", flush=True)
        model = load_checkpoint(ck, device).eval()
        report["sources"][name] = summarise(
            evaluate_source(session, _model_loader(model, session, device), enc))
        report["sources"][name]["checkpoint"] = str(ck)

    out = Path(ROOT) / args.out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[rcs] wrote {out}")
    for src, s in report["sources"].items():
        o = s["overall"]
        sd = f"{o['sd']:.4f}" if o["sd"] is not None else "n/a"
        print(f"  {src:34s} n={o['n']:2d}  mean={o['mean']:.4f}  sd={sd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
