"""M9: the ablation ladder on the SEALED TEST split, every rung through one path.

Rungs A (U-Net), B (U-Net + temporal), C (StructFormer), D (StructFormer +
temporal), E (StructFormer + entity conditioning), all trained on ladder-v3 data
(20,790 frames), same seed, objective family, early-stopping and selection rule.
Scored here on the same 3,960 frames (4 held-out sessions, oil_impasto), each with
the inputs it actually runs with: temporal rungs get their real previous frame,
the entity rung gets its engine masks. Scoring a temporal rung frame-by-frame, or
an entity rung without masks, measures a model that does not exist (Decisions 58,
66, 80).

Metrics: L1, PSNR, SSIM, LPIPS and sharpness ratio (sum of Sobel energy of the
prediction over that of the target — Decision 72's primary metric), with the
native engine frame as the do-nothing anchor. Rungs F / F0 (M8) are scored on
two-style clips by `scripts/eval_m8.py`; their oil subset is quoted beside this
table with that caveat rather than mixed into it.

Objective note, stated rather than hidden: B and D train with an additional
temporal-consistency term (the §4 curriculum's temporal stage), so A->B and C->D
differences mix architecture with that term.
"""

from __future__ import annotations

import datetime as dt
import gc
import json

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from eval.metrics.image_quality import LpipsMetric, gradient_energy, lpips_available, psnr, ssim
from renderer.entity.dataset import build_entity_splits
from renderer.temporal.pair_dataset import TemporalFrameDataset
from renderer.training.dataset import target_dir_for
from renderer.training.multi_session import build_splits, load_session_split
from renderer.training.train_baseline import ROOT, load_checkpoint, unpack_batch
from data_pipeline.teacher.structural import StructuralSession

SPLIT = "configs/data/session_split_v2.json"
RUNGS = {"A": ("ladder_v3_a", "plain"), "B": ("ladder_v3_b", "temporal"),
         "C": ("ladder_v3_c", "plain"), "D": ("ladder_v3_d", "temporal"),
         "E": ("ladder_v3_e", "entity")}


def temporal_test(sp):
    parts = []
    for sid in sp["test_sessions"]:
        s = StructuralSession(ROOT / "data_pipeline" / "recorder" / sid)
        parts.append(TemporalFrameDataset(s, target_dir_for(s.session_id, "oil_impasto"), s.frame_ids()))
    return ConcatDataset(parts)


@torch.no_grad()
def score(model, ds, kind, dev, lp):
    acc = {k: [] for k in ("l1", "psnr", "ssim", "lpips", "gp", "gt")}
    for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=6):
        if kind == "temporal":
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            prev, _ = model.encode(b["structure_prev"])
            pred = model(b["structure_curr"], prev, b["motion_curr"], b["mask_curr"])
            y = b["target_curr"]
        else:
            x, y, extra, _ = unpack_batch(b, dev)
            pred = model(x, *extra)
        pred = pred.float().clamp(0, 1); y = y.float()
        acc["l1"].append((pred - y).abs().flatten(1).mean(1).cpu())
        acc["psnr"].append(psnr(pred, y).cpu()); acc["ssim"].append(ssim(pred, y).cpu())
        if lp is not None:
            acc["lpips"].append(lp(pred, y).cpu())
        acc["gp"].append(gradient_energy(pred).cpu()); acc["gt"].append(gradient_energy(y).cpu())
    c = {k: torch.cat(v) for k, v in acc.items() if v}
    return {"n_frames": int(c["l1"].numel()), "l1": float(c["l1"].mean()),
            "psnr_db": float(c["psnr"].mean()), "ssim": float(c["ssim"].mean()),
            "lpips": float(c["lpips"].mean()) if "lpips" in c else None,
            "sharpness": float(c["gp"].sum() / c["gt"].sum())}


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = LpipsMetric(device=dev) if lpips_available() else None
    sp = load_session_split(ROOT / SPLIT)
    ds = {"plain": build_splits("oil_impasto", sp)[2],
          "entity": build_entity_splits("oil_impasto", sp, k_max=8)[2],
          "temporal": temporal_test(sp)}
    n = {k: len(v) for k, v in ds.items()}
    assert len(set(n.values())) == 1, f"rungs would be scored on different frame counts: {n}"
    out = {"purpose": "M9 ablation ladder, sealed test split, every rung through one scoring path.",
           "split": SPLIT, "style": "oil_impasto", "n_frames": n["plain"],
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "objective_note": "B and D add a temporal-consistency term; A->B and C->D mix architecture with it.",
           "rungs": {}}
    for rung, (ck, kind) in RUNGS.items():
        m = load_checkpoint(ROOT / "checkpoints" / ck / "best.pt", dev).eval()
        r = score(m, ds[kind], kind, dev, lp)
        r["params"] = sum(p.numel() for p in m.parameters()); r["inputs"] = kind
        out["rungs"][rung] = r
        print(f"[m9] {rung}: L1 {r['l1']:.6f}  PSNR {r['psnr_db']:.3f}  SSIM {r['ssim']:.4f}  "
              f"LPIPS {r['lpips']:.4f}  sharp {r['sharpness']:.4f}  ({r['params']:,} params)", flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()
    try:
        f = json.load(open(ROOT / "eval/reports/m8_f_vs_f0.json"))["subsets"]["oil_impasto"]
        out["m8_oil_subset_different_protocol"] = {"F": f["F"], "F0": f["F0"],
            "note": "two-style clip protocol, 4-frame clips, 1,956 oil clips; not the frame set above"}
    except Exception:
        pass
    (ROOT / "eval/reports/m9_ladder.json").write_text(json.dumps(out, indent=2) + "\n")
    print("[m9] wrote eval/reports/m9_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
