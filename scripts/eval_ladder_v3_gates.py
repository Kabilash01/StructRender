"""Score one ladder-v3 rung on the SEALED TEST split — gates included.

One model per process, by design: holding three models plus LPIPS plus two
datasets at once got this killed by the OOM reaper twice. The caller loops.

Usage: python -m scripts.eval_ladder_v3_gates <a|c|e|e_seedB|e_seedC>
"""

from __future__ import annotations

import gc
import json
import sys

import torch
from torch.utils.data import ConcatDataset

from renderer.entity.dataset import build_entity_splits
from renderer.training.multi_session import build_splits, load_session_split
from renderer.training.train_baseline import (ROOT, LpipsMetric, evaluate, load_checkpoint,
                                              load_config, lpips_available, style_acquisition)

SPLIT = "configs/data/session_split_v2.json"


def main() -> int:
    name = sys.argv[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = LpipsMetric(device=dev) if lpips_available() else None
    sp = load_session_split(ROOT / SPLIT)
    _, _, te_plain = build_splits("oil_impasto", sp)
    cfg = load_config(ROOT / f"configs/ladder_v3/model_{name}.json")
    # Entity rungs need their masks; scoring them through the plain dataset would
    # silently disable the entity path and measure a model that does not exist.
    ds = build_entity_splits("oil_impasto", sp, k_max=8)[2] if name.startswith("e") else te_plain

    m = load_checkpoint(ROOT / cfg["train"]["checkpoint_dir"] / "best.pt", dev).eval()
    ev = evaluate(m, ds, dev, lpips_metric=lp)
    parts = te_plain.datasets if isinstance(te_plain, ConcatDataset) else [te_plain]
    rows = [style_acquisition(m, s, dev, lpips_metric=lp) for s in parts]
    w = [len(s) for s in parts]
    tot = float(sum(w))
    g = {k: sum(r[k] * x for r, x in zip(rows, w)) / tot for k in rows[0]}

    out = {"model": name, "seed": cfg["train"]["seed"], "split": SPLIT,
           "l1": ev["l1"], "psnr": ev["psnr_db"], "ssim": ev["ssim"],
           "sharp": g["sharpness_ratio"], "anchor": g.get("sharpness_native_anchor")}
    (ROOT / "eval" / "reports" / f"ladder_v3_{name}_gates.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{name:<10} seed {out['seed']}  L1 {out['l1']:.6f}  PSNR {out['psnr']:.3f}  "
          f"sharp {out['sharp']:.4f}  {'PASS' if out['sharp'] >= 0.90 else 'FAIL'}")
    del m
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
