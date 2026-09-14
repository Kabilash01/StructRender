"""Score Models B and D on the sealed test split WITH their history — KB Decision 66.

`scripts/eval_ladder_v2.py` scores every rung through a frame-wise dataset that
supplies no previous frame, so both temporal rungs ran **cold-start** — reduced to
their own baselines. Their numbers there are not the temporal models' and the
artifact says so. This supplies the history and measures what B and D actually do.

Reports the same three gates, plus the cold-start figures side by side, so the
difference the temporal mechanism makes is the visible quantity.

    python -m scripts.eval_temporal_ladder_v2      (needs CUDA)
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data_pipeline.teacher.structural import StructuralSession
from renderer.temporal.pair_dataset import TemporalFrameDataset
from renderer.training.dataset import target_dir_for
from renderer.training.multi_session import load_session_split, split_report
from renderer.training.train_baseline import (ROOT, LpipsMetric, load_checkpoint, load_config,
                                              lpips_available, psnr, ssim)

TEMPORAL = {"b": "configs/ladder_v2/model_b.json", "d": "configs/ladder_v2/model_d.json"}
GRAD_K = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])


def _grad_energy(x: torch.Tensor) -> float:
    """Squared-Sobel energy on luma — the quantity the sharpness gate uses
    (KB Decision 35: mean-absolute Sobel is non-monotone under blur here)."""
    luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    kx = GRAD_K.to(x.device, x.dtype)
    gx = torch.nn.functional.conv2d(luma, kx, padding=1)
    gy = torch.nn.functional.conv2d(luma, kx.transpose(-1, -2), padding=1)
    return float((gx ** 2 + gy ** 2).mean())


def _test_dataset(style_id: str, split: dict):
    root = ROOT / "data_pipeline" / "recorder"
    parts = []
    for sid in split["test_sessions"]:
        s = StructuralSession(root / sid)
        parts.append(TemporalFrameDataset(s, target_dir_for(s.session_id, style_id), s.frame_ids()))
    return ConcatDataset(parts)


@torch.no_grad()
def evaluate_with_history(model, ds, device, lp=None, use_history: bool = True) -> dict:
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    n = l1 = ps = ss = lpv = 0.0
    ge_pred = ge_targ = 0.0
    for b in loader:
        sp = b["structure_prev"].to(device)
        sc = b["structure_curr"].to(device)
        mo = b["motion_curr"].to(device)
        mk = b["mask_curr"].to(device)
        y = b["target_curr"].to(device)
        if use_history:
            feats, _ = model.encode(sp)
            out = model(sc, feats, mo, mk).clamp(0, 1)
        else:
            # Cold start through the IDENTICAL code path, so the only difference
            # between the two rows is the history — not the metric implementation.
            # Reading a delta across two different sharpness computations, as an
            # earlier version of this script did, measures the code not the model.
            out = model(sc).clamp(0, 1)
        k = sc.shape[0]
        n += k
        l1 += float(torch.nn.functional.l1_loss(out, y)) * k
        # psnr/ssim return PER-IMAGE [B] tensors, not scalars — sum them so the
        # mean below is over frames, exactly as `evaluate()` does.
        ps += float(psnr(out, y).sum())
        ss += float(ssim(out, y).sum())
        if lp is not None:
            _l = lp(out, y)
            lpv += float(_l.sum()) if torch.is_tensor(_l) and _l.numel() > 1 else float(_l) * k
        ge_pred += _grad_energy(out) * k
        ge_targ += _grad_energy(y) * k
    return {"n_frames": int(n), "l1": l1 / n, "psnr_db": ps / n, "ssim": ss / n,
            "lpips_alex": (lpv / n) if lp is not None else None,
            "sharpness_ratio": ge_pred / ge_targ}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_session_split()
    ds = _test_dataset("oil_impasto", split)
    lp = LpipsMetric(device=device) if lpips_available() else None
    cold = json.loads((ROOT / "eval" / "reports" / "ladder_v2_test.json").read_text())

    report = {
        "purpose": ("Models B and D scored on the sealed test split WITH previous-frame history. "
                    "ladder_v2_test.json scored them cold-start, reduced to their baselines "
                    "(KB Decision 66)."),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__, "style_id": "oil_impasto",
        "split": split_report(split),
        "n_test_frames": len(ds),
        "models": {},
    }
    for name, cfg_path in TEMPORAL.items():
        cfg = load_config(ROOT / cfg_path)
        ck = ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
        if not ck.is_file():
            report["models"][name] = {"skipped": str(ck)}
            continue
        print(f"[eval] model {name.upper()} with history ...", flush=True)
        model = load_checkpoint(ck, device).eval()
        hist = evaluate_with_history(model, ds, device, lp, use_history=True)
        cold_same = evaluate_with_history(model, ds, device, lp, use_history=False)
        cs = cold["models"][name]
        report["models"][name] = {
            "ladder_model": name.upper(), "arch": cfg["model"]["arch"],
            "with_history": hist,
            "cold_start_same_path": cold_same,
            "cold_start_from_frame_eval": {
                "l1": cs["test_set"]["l1"], "psnr_db": cs["test_set"]["psnr_db"],
                "sharpness_ratio": cs["gates"]["sharpness_ratio"],
                "note": ("From ladder_v2_test.json, whose sharpness uses style_acquisition's "
                         "implementation. NOT comparable to the rows above — kept only for "
                         "cross-reference."),
            },
            "delta_from_history": {
                "l1_pct": 100 * (hist["l1"] - cold_same["l1"]) / cold_same["l1"],
                "psnr_db": hist["psnr_db"] - cold_same["psnr_db"],
                "sharpness": hist["sharpness_ratio"] - cold_same["sharpness_ratio"],
                "note": "Both rows from the SAME code path; the only difference is the history.",
            },
            "sharpness_pass": hist["sharpness_ratio"] >= 0.90,
            "baseline_for_comparison": "A" if name == "b" else "C",
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = ROOT / "eval" / "reports" / "ladder_v2_temporal_test.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote {out}\n")
    print("same code path, only the history differs:")
    print(f"{'model':<7}{'cold L1':>11}{'hist L1':>11}{'dL1%':>8}{'cold sharp':>12}{'hist sharp':>12}{'d':>9}")
    for n, e in report["models"].items():
        if "with_history" not in e:
            continue
        c, h = e["cold_start_same_path"], e["with_history"]
        print(f"{n.upper():<7}{c['l1']:>11.6f}{h['l1']:>11.6f}"
              f"{e['delta_from_history']['l1_pct']:>+8.1f}"
              f"{c['sharpness_ratio']:>12.4f}{h['sharpness_ratio']:>12.4f}"
              f"{e['delta_from_history']['sharpness']:>+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
