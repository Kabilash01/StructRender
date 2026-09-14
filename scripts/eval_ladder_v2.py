"""Score every ladder-v2 rung on the SEALED TEST split — KB Decision 64.

The test sessions were fixed in `configs/data/session_split_v1.json` by position
in the sorted session-id list, before any model existed and before any metric was
computed. Nothing has trained on them and no checkpoint was selected on them, so
this is the first independent generalisation estimate in the project: every
earlier number was a best-of-N-epochs minimum on the frames used to pick the
checkpoint.

Reports val (selection) and test side by side, because the gap between them is
the thing no previous measurement could show.

    python -m scripts.eval_ladder_v2
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import torch

from torch.utils.data import ConcatDataset

from renderer.training.multi_session import build_splits, load_session_split, split_report
from renderer.training.train_baseline import (ROOT, LpipsMetric, evaluate, load_checkpoint,
                                              load_config, lpips_available, objective_string,
                                              parameter_count, style_acquisition)

MODELS = {m: f"configs/ladder_v2/model_{m}.json" for m in ("a", "b", "c", "d")}


def style_acquisition_multi(model, ds, device, lpips_metric=None) -> dict:
    """`style_acquisition` needs `dataset.session` for the native reference, which
    a ConcatDataset does not have. Run it per sub-dataset and average, weighting
    each by its frame count so sessions of unequal length do not skew the mean."""
    parts = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
    rows, weights = [], []
    for sub in parts:
        rows.append(style_acquisition(model, sub, device, lpips_metric=lpips_metric))
        weights.append(len(sub))
    total = float(sum(weights))
    return {k: sum(r[k] * w for r, w in zip(rows, weights)) / total for k in rows[0]}


def _temporal_wrap(model, ds):
    """Model B and D need history. The eval datasets are frame-wise, so a
    temporal rung scored here runs COLD-START — identical to its baseline. That
    would silently measure the wrong model, so these are marked rather than
    quietly scored."""
    return hasattr(model, "temporal_fuse")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = load_session_split()
    _train, val_ds, test_ds = build_splits("oil_impasto", sp)
    lp = LpipsMetric(device=device) if lpips_available() else None

    report = {
        "purpose": ("Ladder v2 scored on the sealed test split — the project's first independent "
                    "generalisation estimate (KB Decision 64)."),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "style_id": "oil_impasto",
        "split": split_report(sp),
        "gates": {"min_sharpness_ratio": 0.90, "min_hist_travel_fraction_of_target": 0.80,
                  "note": "Imported unchanged from M4/M5 (Decisions 35, 41). Not re-tuned."},
        "models": {},
    }

    for name, cfg_path in MODELS.items():
        cfg = load_config(ROOT / cfg_path)
        ck = ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
        if not ck.is_file():
            report["models"][name] = {"skipped": f"no checkpoint at {ck}"}
            continue
        print(f"[eval] model {name.upper()} ...", flush=True)
        model = load_checkpoint(ck, device).eval()
        entry = {
            "ladder_model": name.upper(),
            "arch": cfg["model"]["arch"],
            "checkpoint": str(ck.relative_to(ROOT)),
            "parameters": parameter_count(model),
            "objective": objective_string(cfg),
            "selection_set_val": evaluate(model, val_ds, device, lpips_metric=lp),
            "test_set": evaluate(model, test_ds, device, lpips_metric=lp),
            "style_acquisition_test": style_acquisition_multi(model, test_ds, device, lpips_metric=lp),
        }
        if _temporal_wrap(model, test_ds):
            entry["temporal_caveat"] = (
                "TEMPORAL RUNG SCORED COLD-START. The frame-wise eval supplies no previous-frame "
                "history, so this model ran reduced to its own baseline (Model A for B, Model C "
                "for D). These numbers are NOT the temporal model's; a history-aware evaluation "
                "is required before any temporal claim is made from them.")
        # gates
        sa = entry["style_acquisition_test"]
        hist_ratio = sa["hist_pred_vs_native"] / sa["hist_target_vs_native"]
        entry["gates"] = {
            "sharpness_ratio": sa["sharpness_ratio"],
            "sharpness_pass": sa["sharpness_ratio"] >= 0.90,
            "sharpness_native_anchor": sa["sharpness_ratio_native"],
            "hist_travel_ratio": hist_ratio,
            "hist_pass": hist_ratio >= 0.80,
            "psnr_vs_do_nothing": sa["psnr_pred_vs_target_db"] - sa["psnr_native_vs_target_db"],
            "beats_do_nothing": sa["psnr_pred_vs_target_db"] > sa["psnr_native_vs_target_db"],
        }
        entry["generalisation_gap"] = {
            "val_l1": entry["selection_set_val"]["l1"], "test_l1": entry["test_set"]["l1"],
            "gap_pct": 100 * (entry["test_set"]["l1"] - entry["selection_set_val"]["l1"])
                       / entry["selection_set_val"]["l1"],
            "note": "How much worse the model is on data it was never selected on.",
        }
        report["models"][name] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = ROOT / "eval" / "reports" / "ladder_v2_test.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote {out}\n")
    print(f"{'model':<8}{'test_l1':>10}{'test_psnr':>11}{'sharp':>9}{'hist':>8}{'gap%':>8}")
    for n, e in report["models"].items():
        if "gates" not in e:
            continue
        g = e["gates"]
        print(f"{n.upper():<8}{e['test_set']['l1']:>10.6f}{e['test_set']['psnr_db']:>11.3f}"
              f"{g['sharpness_ratio']:>9.4f}{g['hist_travel_ratio']:>8.3f}"
              f"{e['generalisation_gap']['gap_pct']:>+8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
