"""M8 exit criterion 4: does EPAM memory improve over the no-memory model?

Model F (gamma 0.9) vs Model F0 (gamma 0) — identical code, data, steps and
selection rule, so the difference is memory alone (KB Decision 78). Scored on the
SEALED TEST sessions, both styles, through the same clip path and the same
`memory_step` the runtime uses.

Primary metric, pre-registered in Decision 72: sharpness ratio (mean Sobel
gradient energy of prediction / of target). Also reported: L1, sharpness on
ENTITY pixels (where memory acts), and the subset of clips where memory CAN act
(last-frame entity also seen in the clip's history).

Noise yardstick: one seed per model, so run-to-run spread is taken from the
three-seed measurements in Decision 77 (sharpness sd 0.0067 for E, 0.0079 for C).
"""

from __future__ import annotations

import datetime as dt
import json
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval.metrics.image_quality import gradient_energy
from renderer.epam.clips import EPAMClipDataset
from renderer.epam.train_epam import clip_forward, load_f
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT

SEED_SD_SHARP = 0.0073   # mean of E (0.0067) and C (0.0079) three-seed sd, Decision 77


def energy_map(img):
    """Per-pixel squared Sobel energy of luma — the same operator as gradient_energy."""
    w = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)
    luma = (img * w).sum(1, keepdim=True)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device).view(1, 1, 3, 3)
    p = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(p, kx), F.conv2d(p, kx.transpose(2, 3))
    return (gx * gx + gy * gy)[:, 0]


@torch.no_grad()
def score(tag, ds, device):
    model, gamma, tau = load_f(ROOT / f"checkpoints/epam_{tag}/best.pt", device)
    rows = []
    for b in DataLoader(ds, batch_size=4, shuffle=False, num_workers=2):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = clip_forward(model, b, gamma, tau, device, grad_last=False)
        pred = pred.float().clamp(0, 1); y = b["target"].to(device)
        gp, gt = gradient_energy(pred), gradient_energy(y)
        ent = (b["masks"][:, -1].to(device).sum(1) > 0)                 # [B,H,W]
        ep, et = energy_map(pred), energy_map(y)
        keys = b["keys"]
        for i in range(pred.shape[0]):
            last = set(keys[i, -1][keys[i, -1] > 0].tolist())
            hist = set(keys[i, :-1][keys[i, :-1] > 0].tolist())
            m = ent[i]
            rows.append({
                "style": int(b["style"][i]), "l1": float((pred[i] - y[i]).abs().mean()),
                "gp": float(gp[i]), "gt": float(gt[i]),
                "egp": float(ep[i][m].mean()) if m.sum() > 50 else None,
                "egt": float(et[i][m].mean()) if m.sum() > 50 else None,
                "memory_can_act": bool(last & hist)})
    del model; torch.cuda.empty_cache()
    return rows


def summarise(rows, pick=lambda r: True):
    r = [x for x in rows if pick(x)]
    e = [x for x in r if x["egp"] is not None]
    return {"n": len(r),
            "l1": sum(x["l1"] for x in r) / len(r),
            "sharpness": sum(x["gp"] for x in r) / sum(x["gt"] for x in r),
            "entity_sharpness": (sum(x["egp"] for x in e) / sum(x["egt"] for x in e)) if e else None,
            "n_entity": len(e)}


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = load_session_split(ROOT / "configs/data/session_split_v2.json")
    styles = ["oil_impasto", "watercolor"]
    ds = EPAMClipDataset(ROOT, sp["test_sessions"], styles, train=False, strides=(1, 2, 4))
    res = {t: score(t, ds, dev) for t in ("f", "f0")}
    subsets = {
        "all": lambda r: True,
        "oil_impasto": lambda r: r["style"] == 0,
        "watercolor": lambda r: r["style"] == 1,
        "memory_can_act": lambda r: r["memory_can_act"],
    }
    out = {"purpose": "M8 criterion 4: Model F (memory) vs F0 (no-memory control) on the sealed test split.",
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "test_sessions": sp["test_sessions"], "n_clips": len(ds),
           "primary_metric": "sharpness ratio (Decision 72)",
           "noise_yardstick": {"seed_sd_sharpness": SEED_SD_SHARP,
                               "two_sigma_for_one_run_each": 2 * math.sqrt(2) * SEED_SD_SHARP,
                               "source": "Decision 77 three-seed spreads; F and F0 have one seed each"},
           "subsets": {}}
    thr = 2 * math.sqrt(2) * SEED_SD_SHARP
    for name, pick in subsets.items():
        a, b = summarise(res["f"], pick), summarise(res["f0"], pick)
        d = a["sharpness"] - b["sharpness"]
        out["subsets"][name] = {"F": a, "F0": b, "sharpness_F_minus_F0": d,
                                "l1_F_vs_F0_pct": 100 * (a["l1"] - b["l1"]) / b["l1"],
                                "beyond_2sigma_seed_noise": abs(d) > thr}
    out["verdict"] = ("memory improves sharpness beyond seed noise"
                      if out["subsets"]["all"]["beyond_2sigma_seed_noise"]
                      and out["subsets"]["all"]["sharpness_F_minus_F0"] > 0
                      else "no difference distinguishable from seed noise"
                      if not out["subsets"]["all"]["beyond_2sigma_seed_noise"]
                      else "memory makes sharpness WORSE beyond seed noise")
    p = ROOT / "eval/reports/m8_f_vs_f0.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}  ({len(ds)} clips)\n")
    print(f"{'subset':<16}{'n':>6}{'F sharp':>10}{'F0 sharp':>10}{'diff':>9}{'F L1':>10}{'F0 L1':>10}{'L1 %':>8}")
    for k, v in out["subsets"].items():
        print(f"{k:<16}{v['F']['n']:>6}{v['F']['sharpness']:>10.4f}{v['F0']['sharpness']:>10.4f}"
              f"{v['sharpness_F_minus_F0']:>+9.4f}{v['F']['l1']:>10.6f}{v['F0']['l1']:>10.6f}{v['l1_F_vs_F0_pct']:>+8.2f}")
    a = out["subsets"]["all"]
    print(f"\nentity-pixel sharpness: F {a['F']['entity_sharpness']:.4f}  F0 {a['F0']['entity_sharpness']:.4f}"
          f"  (n={a['F']['n_entity']})")
    print(f"2-sigma seed noise (one run each): {thr:.4f}")
    print(f"VERDICT: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
