"""Train Model F (EPAM) or its no-memory control — identical except `gamma`.

    python -m renderer.epam.train_epam --config configs/epam/model_f.json
    python -m renderer.epam.train_epam --config configs/epam/model_f0.json

Both resume from Model E (§4 curriculum, stage 5 from stage 4), train on the same
clips, same styles, same objective (1*L1 + 0.5*LPIPS, Decision 41), same schedule,
same early-stopping and selection rule (lowest val_l1, Decisions 35/62). gamma is
the only difference, so F - F0 measures memory and nothing else.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from renderer.epam.clips import EPAMClipDataset
from renderer.epam.model import StructRenderF
from renderer.epam.runtime import memory_step
from renderer.losses.perceptual import PerceptualReconstructionLoss
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, assert_finite_loss, load_config


def model_kwargs(cfg: dict) -> dict:
    m = cfg["model"]
    return dict(image_size=tuple(m["image_size"]), embed_dim=m["embed_dim"],
                depths=tuple(m["depths"]), num_heads=tuple(m["num_heads"]),
                window_sizes=tuple(m["window_sizes"]))


def build(cfg: dict) -> StructRenderF:
    return StructRenderF(cfg["data"]["styles"], k_max=cfg["model"]["k_max"], **model_kwargs(cfg))


def clip_forward(model, b, gamma, tau, device, grad_last=True):
    """Run memory over the clip's history frames (no grad — memory is detached,
    §4), then render the last frame with grad. Returns the final prediction."""
    S = b["structure"].to(device); M = b["masks"].to(device)
    K = b["keys"]; P = b["px"]; sty = b["style"].to(device)
    T = S.shape[1]
    stores = [dict() for _ in range(S.shape[0])]
    with torch.no_grad():
        for t in range(T - 1):
            f, _ = model.encode(S[:, t], sty)
            m = model.bottleneck_masks(M[:, t], f)
            memory_step(stores, K[:, t], model.describe(f, m), P[:, t], gamma, tau)
    ctx = torch.enable_grad() if grad_last else torch.no_grad()
    with ctx:
        f, skip = model.encode(S[:, -1], sty)
        m = model.bottleneck_masks(M[:, -1], f)
        mem = memory_step(stores, K[:, -1], model.describe(f, m), P[:, -1], gamma, tau)
        return model.decode(f, skip, mem, m)


@torch.no_grad()
def evaluate(model, ds, gamma, tau, device, lpips=None, batch_size=4):
    model.eval()
    l1s, lps = [], []
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = clip_forward(model, b, gamma, tau, device, grad_last=False)
        pred = pred.float().clamp(0, 1); y = b["target"].to(device)
        l1s.append((pred - y).abs().flatten(1).mean(1).cpu())
        if lpips is not None:
            lps.append(lpips(pred, y).cpu())
    out = {"val_l1": float(torch.cat(l1s).mean()), "val_n": len(ds)}
    if lps:
        out["val_lpips"] = float(torch.cat(lps).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test only")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    t, e = cfg["train"], cfg["epam"]
    gamma, tau = float(e["gamma"]), float(e["tau"])
    torch.manual_seed(t["seed"]); np.random.seed(t["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = load_session_split(ROOT / cfg["data"]["split_file"])
    styles = cfg["data"]["styles"]
    tr = EPAMClipDataset(ROOT, sp["train_sessions"], styles, train=True,
                         clip_len=e["clip_len"], strides=tuple(e["strides"]))
    va = EPAMClipDataset(ROOT, sp["val_sessions"], styles, train=False,
                         clip_len=e["clip_len"], strides=tuple(e["strides"]), every=e["val_every"])
    loader = DataLoader(tr, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                        num_workers=t["num_workers"], persistent_workers=True)

    model = build(cfg).to(dev)
    src = torch.load(ROOT / cfg["init_from"], map_location="cpu", weights_only=False)
    info = model.load_from_model_e(src["model"])
    if info["skipped"] or info["missing"]:
        raise SystemExit(f"resume from Model E incomplete: {info}")
    print(json.dumps({"init_from": cfg["init_from"], "loaded": info["loaded"], "gamma": gamma,
                      "tau": tau, "train_clips": len(tr), "val_clips": len(va)}), flush=True)

    from renderer.training.train_baseline import LpipsMetric, lpips_available
    lp = LpipsMetric(device=dev) if lpips_available() else None
    L = cfg["loss"]
    crit = PerceptualReconstructionLoss(w_l1=L["l1"], w_perceptual=L["perceptual"],
                                        net=L["perceptual_net"]).to(dev)
    if L["perceptual"] > 0 and not crit.used_perceptual:
        raise RuntimeError("perceptual loss requested but lpips unavailable (Decision 41)")

    accum = int(t["accum_steps"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = math.ceil(len(loader) / accum)
    total = t["epochs"] * steps_per_epoch
    warm = int(t.get("warmup_epochs", 0)) * steps_per_epoch
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, total_iters=max(1, warm)),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total - warm))],
        milestones=[max(1, warm)])

    ck = ROOT / t["checkpoint_dir"]; ck.mkdir(parents=True, exist_ok=True)
    es = t["early_stopping"]; best = {"val_l1": float("inf"), "epoch": -1}
    stale, hist, t0, stopped = 0, [], time.time(), None
    # epoch -1: the resumed model before any EPAM training, so the gain is visible
    v = evaluate(model, va, gamma, tau, dev, lp); v["epoch"] = -1
    print(json.dumps(v), flush=True); hist.append(v)
    step = 0
    for ep in range(t["epochs"]):
        model.train(); run, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = clip_forward(model, b, gamma, tau, dev)
                loss, _ = crit(pred.float(), b["target"].to(dev))
            val = assert_finite_loss(loss, step=i, epoch=ep)
            (loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += val; n += 1; step += 1
            if a.max_steps and step >= a.max_steps:
                print(json.dumps({"smoke_ok": True, "steps": step, "loss": run / n,
                                  "peak_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
                return 0
        rec = {"epoch": ep, "train_loss": run / max(1, n)}
        if (ep + 1) % t["eval_every_epochs"] == 0 or ep == t["epochs"] - 1:
            rec.update(evaluate(model, va, gamma, tau, dev, lp))
            if rec["val_l1"] < best["val_l1"] * (1 - es["min_rel_delta"]):
                stale = 0
            else:
                stale += 1
            if rec["val_l1"] < best["val_l1"]:
                best = {"val_l1": rec["val_l1"], "epoch": ep}
                torch.save({"arch": "structrender_f", "cfg": cfg, "gamma": gamma, "tau": tau,
                            "model": model.state_dict(), "epoch": ep}, ck / "best.pt")
        print(json.dumps(rec), flush=True); hist.append(rec)
        if stale >= es["patience_evals"]:
            stopped = {"stopped_at_epoch": ep, "best_epoch": best["epoch"]}
            print(json.dumps({"early_stop": stopped}), flush=True)
            break
    rep = {"config": a.config, "gamma": gamma, "tau": tau, "best": best, "stopped_early": stopped,
           "wall_seconds": time.time() - t0, "history": hist,
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (ROOT / cfg["report"]).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


def load_f(path, device) -> tuple[StructRenderF, float, float]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["cfg"]).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m, float(ck["gamma"]), float(ck["tau"])


if __name__ == "__main__":
    raise SystemExit(main())
