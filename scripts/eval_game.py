"""Score a game-trained rung on the game's SEALED TEST sessions (Decisions 87, 90).

Image quality (L1, PSNR, SSIM, LPIPS, sharpness ratio) with the model's real
previous frame, and temporal error (flicker) against the native engine and the
teacher targets on the same pairs. Test sessions were fixed by position before
any target or model existed (configs/data/game_split_v1_oil.json).
"""
from __future__ import annotations
import datetime as dt, json, sys
import numpy as np, torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.image_quality import LpipsMetric, gradient_energy, lpips_available, psnr, ssim
from eval.metrics.temporal import engine_confidence, evaluate_source, summarise
from renderer.temporal.pair_dataset import TemporalFrameDataset
from renderer.temporal.pairs import FramePair, consecutive_pairs
from renderer.training.dataset import build_input, target_dir_for
from renderer.training.train_baseline import ROOT, load_checkpoint

def main(ckpt="checkpoints/game_b_oil/best.pt", split="configs/data/game_split_v1_oil.json", style="oil_impasto") -> int:
    dev = torch.device("cuda"); lp = LpipsMetric(device=dev) if lpips_available() else None
    sp = json.load(open(ROOT / split)); m = load_checkpoint(ROOT / ckpt, dev).eval()
    sess = [StructuralSession(ROOT / "data_pipeline/recorder" / s) for s in sp["test_sessions"]]
    ds = ConcatDataset([TemporalFrameDataset(s, target_dir_for(s.session_id, style), s.frame_ids()) for s in sess])
    acc = {k: [] for k in ("l1", "psnr", "ssim", "lpips", "gp", "gt", "nl1", "gn")}
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=4, num_workers=6):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            prev, _ = m.encode(b["structure_prev"])
            p = m(b["structure_curr"], prev, b["motion_curr"], b["mask_curr"]).float().clamp(0, 1); y = b["target_curr"].float()
            acc["l1"].append((p - y).abs().flatten(1).mean(1).cpu()); acc["psnr"].append(psnr(p, y).cpu()); acc["ssim"].append(ssim(p, y).cpu())
            if lp is not None: acc["lpips"].append(lp(p, y).cpu())
            acc["gp"].append(gradient_energy(p).cpu()); acc["gt"].append(gradient_energy(y).cpu())
    c = {k: torch.cat(v) for k, v in acc.items() if v}
    img = {"n_frames": int(c["l1"].numel()), "l1": float(c["l1"].mean()), "psnr_db": float(c["psnr"].mean()),
           "ssim": float(c["ssim"].mean()), "lpips": float(c["lpips"].mean()) if "lpips" in c else None,
           "sharpness": float(c["gp"].sum() / c["gt"].sum())}
    te = {"native": [], "target": [], "model": []}
    for s in sess:
        tdir = target_dir_for(s.session_id, style)
        idx = {int(json.loads(l)["frame_id"]): json.loads(l)["target_rgb_path"] for l in (tdir / "manifest.jsonl").read_text().splitlines() if l.strip()}
        tl = lambda f, _t=tdir, _i=idx: np.asarray(Image.open(_t / _i[f]).convert("RGB"), dtype=np.float32) / 255.0
        pairs = [q for q in consecutive_pairs(s, s.frame_ids()) if q.tick_delta == 1]
        masks = {q.curr_id: engine_confidence(s, q) for q in pairs}
        ids = set(s.frame_ids())
        def ml(f, _s=s, _ids=ids):
            x = torch.from_numpy(build_input(_s.load_structure(f)))[None].to(dev)
            with torch.no_grad():
                if f - 1 in _ids and _s.motion_valid(f):
                    pf, _ = m.encode(torch.from_numpy(build_input(_s.load_structure(f - 1)))[None].to(dev))
                    cf = engine_confidence(_s, FramePair(prev_id=f - 1, curr_id=f, tick_delta=1))
                    y = m(x, pf, cf["motion"].to(dev), cf["mask"].to(dev))
                else:
                    y = m(x)
            return np.transpose(y[0].float().clamp(0, 1).cpu().numpy(), (1, 2, 0))
        nat = summarise(evaluate_source(s, lambda f, _s=s: _s.load_native_rgb(f).astype(np.float32) / 255.0, pairs, tl, masks))
        moving = {q["curr_id"] for q in nat["per_pair"] if (q.get("median_motion_px") or 0) >= 1.0}
        for name, fn in (("native", lambda f, _s=s: _s.load_native_rgb(f).astype(np.float32) / 255.0), ("target", tl), ("model", ml)):
            r = summarise(evaluate_source(s, fn, pairs, tl, masks))
            te[name] += [q["te"] for q in r["per_pair"] if q["curr_id"] in moving]
    tem = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in te.items()}
    rep = {"checkpoint": ckpt, "split": split, "style": style, "test_sessions": sp["test_sessions"],
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "image_quality": img, "temporal_error": tem}
    out = ROOT / "eval/reports/game_b_oil_test.json"; out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(img)); print({k: round(v["mean"], 6) for k, v in tem.items()}, "n =", tem["model"]["n"]); print("wrote", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
