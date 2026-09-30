"""Train / overfit ladder **Model D** — the temporal StructFormer, KB §10 Milestone 6.

    python -m renderer.training.train_temporal --config configs/temporal/m6_temporal.json --mode overfit
    python -m renderer.training.train_temporal --config configs/temporal/m6_temporal.json --mode train

**Why this is a separate entry point and why that is not a fork.** KB Decision 35's
rule is that every §11 rung shares the split, the selection rule and the
objective, and that a forked training script is how two rungs silently drift
apart on one of the three. Model D's *step* genuinely cannot be Model C's — it
consumes three frames, a motion field and a confidence mask where Model C
consumes one tensor — but everything Decision 35 protects is **imported** from
:mod:`renderer.training.train_baseline` rather than copied: `resolve_data`
(session, targets, split), `report_path` / `assert_reports_resolvable`
(Decision 52), `assert_finite_loss` (§11 gate item 3) and the reconstruction
criterion itself. `configs/temporal/tests` asserts the config blocks match Model
C's field by field, so a drift fails a test rather than reaching a comparison.

**The training step, and the previous-feature decision it encodes.** Per sample
``(t-2, t-1, t)``::

    enc_{t-2} = encode(S_{t-2})                                  # no grad: history only
    enc_{t-1} = encode(S_{t-1})                                  # history for t, WITH grad
    pred_{t-1} = decode(fuse(enc_{t-1}, enc_{t-2}, V_{t-1}, M_{t-1}))   # no grad: reference only
    enc_t     = encode(S_t)
    pred_t    = decode(fuse(enc_t, enc_{t-1}, V_t, M_t))

    L = 1.0*L1(pred_t, I_t) + 0.5*LPIPS(pred_t, I_t) + w_temp * L_temp(pred_t, pred_{t-1})

Two properties are deliberate:

1. **The history is `encode(S_{t-1})` — features of the previous frame's
   *structural buffers*, never of a rendered or teacher image.** At inference the
   engine supplies ``S_{t-1}`` exactly as the dataset does here, so this is not
   teacher forcing with an exposure gap: training and inference compute the
   identical function of quantities both have. It is also the same rule the KB
   fixes for entity features — pooled from the encoder on the deterministic
   G-buffer, never from generated RGB.
2. **`pred_{t-1}` is produced under `no_grad`**, so the temporal term only ever
   pushes ``pred_t`` toward the (already emitted, immutable) previous frame. A
   symmetric term could be minimised by corrupting the *past* frame instead,
   which is a degenerate solution the causal direction rules out for free — and
   it makes that whole branch activation-free.

Reconstruction is supervised on ``pred_t`` only, once per frame per epoch, so
samples-seen-per-epoch stays comparable with Models A and C (KB Decision 51's
protocol) instead of doubling.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.metrics.image_quality import LpipsMetric, lpips_available, psnr, ssim
from renderer.losses.temporal import TemporalObjective
from renderer.structformer.model import parameter_count
from renderer.temporal.pair_dataset import TemporalFrameDataset, TemporalTripleDataset
from renderer.temporal.temporal_model import TemporalStructFormer
from renderer.training.train_baseline import (
    ROOT,
    _amp_dtype,
    assert_finite_loss,
    assert_reports_resolvable,
    load_config,
    report_path,
    resolve_data,
)

ARCH = "temporal_structformer"   # default, and the arch of every pre-Decision-80 run

#: Both temporal rungs train here, because both need the previous-frame history
#: this trainer supplies. Model B used to go through `train_baseline`, which has no
#: temporal data path, so its fusion never received an input (KB Decision 80).
MODEL_KEYS_BY_ARCH = {
    "temporal_structformer": (
        "in_channels", "out_channels", "image_size", "embed_dim", "depths",
        "num_heads", "window_sizes", "style_dim", "attn_impl", "shift_impl",
    ),
    "temporal_unet": ("in_channels", "out_channels", "base_channels", "image_size"),
    # Two-style Model B (KB Decision 94): TemporalUNet + per-style decoder FiLM.
    "style_temporal_unet": ("in_channels", "out_channels", "base_channels", "image_size",
                            "style_ids"),
}
MODEL_KEYS = MODEL_KEYS_BY_ARCH[ARCH]


def arch_of(cfg: dict) -> str:
    a = cfg["model"].get("arch", ARCH)
    if a not in MODEL_KEYS_BY_ARCH:
        raise KeyError(f"{__name__} trains {sorted(MODEL_KEYS_BY_ARCH)}, not {a!r}")
    return a


def model_kwargs(cfg: dict) -> dict:
    m = cfg["model"]
    kw = {k: m[k] for k in MODEL_KEYS_BY_ARCH[arch_of(cfg)] if k in m}
    for k in ("image_size", "depths", "num_heads", "window_sizes"):
        if k in kw:
            kw[k] = tuple(kw[k])
    return kw


def build_model(cfg: dict):
    if arch_of(cfg) == "temporal_unet":
        from renderer.temporal.temporal_unet import TemporalUNet
        return TemporalUNet(**model_kwargs(cfg))
    if arch_of(cfg) == "style_temporal_unet":
        from renderer.temporal.style_unet import StyleTemporalUNet
        if list(cfg["model"]["style_ids"]) != list(cfg["data"]["styles"]):
            raise ValueError("model.style_ids must equal data.styles, in order — the index IS the style")
        return StyleTemporalUNet(**model_kwargs(cfg))
    return TemporalStructFormer(**model_kwargs(cfg))


def init_from_checkpoint(model: TemporalStructFormer, path: str | Path) -> dict:
    """Resume Model D from Model C's best checkpoint — KB Decision 19 / §4
    *Training Curriculum Controller*: "stages reuse checkpoints".

    Loaded non-strictly, but the tolerance is **checked, not assumed**: the only
    parameters allowed to be missing are the temporal fusion's, and nothing in
    the checkpoint may be left unused. A silently-partial load is how a "resumed"
    stage becomes a from-scratch run wearing a checkpoint's name.
    
    NOTE (KB Decision 56): `train.init_from` is null in the shipped M6 config.
    Resuming from Model C is a warm RESTART, not a fine-tune, and it would also
    give Model D 120 epochs of optimisation against Model C's 60. This function
    remains for the staged-training path Decision 19 describes; it is not used
    by the committed Model D run.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    expected_missing = {n for n, _ in model.temporal_fuse.named_parameters(prefix="temporal_fuse")}
    missing = set(incompatible.missing_keys)
    if missing != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"init_from {path} did not load Model C cleanly: missing={sorted(missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}; expected exactly "
            f"{sorted(expected_missing)} to be new"
        )
    return {
        "init_from": str(path),
        "init_from_arch": ckpt.get("arch"),
        "init_from_epoch": ckpt.get("epoch"),
        "init_from_config_version": ckpt.get("config_version"),
        "new_parameters": sorted(expected_missing),
    }


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def _to(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _style(b: dict) -> tuple:
    """Extra decode/forward args: the style index for two-style B, else nothing."""
    return (b["style_idx"],) if "style_idx" in b else ()


def training_forward(model: TemporalStructFormer, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """``(pred_t, pred_{t-1})`` for one triple batch. See the module docstring."""
    with torch.no_grad():
        enc_prev2, _ = model.encode(b["structure_prev2"])
    enc_prev, skip_prev = model.encode(b["structure_prev"])
    with torch.no_grad():
        pred_prev = model.decode(
            model.fuse(enc_prev, enc_prev2, b["motion_prev"], b["mask_prev"]), skip_prev, *_style(b)
        )
    enc, skip = model.encode(b["structure_curr"])
    pred = model.decode(
        model.fuse(enc, enc_prev, b["motion_curr"], b["mask_curr"]), skip, *_style(b)
    )
    return pred, pred_prev


def eval_forward(model: TemporalStructFormer, b: dict) -> torch.Tensor:
    """One frame with its own single-frame history — the M6 gate's rendering rule."""
    enc_prev, _ = model.encode(b["structure_prev"])
    return model(b["structure_curr"], enc_prev, b["motion_curr"], b["mask_curr"], *_style(b))


@torch.no_grad()
def evaluate(model, dataset, device, batch_size: int = 2, lpips_metric=None) -> dict:
    """Model C's `evaluate` on Model D's data contract — same metrics, same rule.

    The selection rule is unchanged and unchangeable here: `run_training`
    checkpoints on the lowest ``l1`` this returns, exactly as Models A and C were
    selected (KB Decision 35).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    l1s, psnrs, ssims, lps = [], [], [], []
    for batch in loader:
        b = _to(batch, device)
        pred = eval_forward(model, b).float()
        y = b["target_curr"]
        l1s.append((pred - y).abs().flatten(1).mean(dim=1).cpu())
        psnrs.append(psnr(pred, y).cpu())
        ssims.append(ssim(pred, y).cpu())
        if lpips_metric is not None:
            lps.append(lpips_metric(pred, y).cpu())
    out = {
        "n_frames": len(dataset),
        "l1": float(torch.cat(l1s).mean()),
        "psnr_db": float(torch.cat(psnrs).mean()),
        "ssim": float(torch.cat(ssims).mean()),
    }
    if lps:
        out["lpips_alex"] = float(torch.cat(lps).mean())
    return out


def build_criterion(cfg: dict, device: torch.device) -> TemporalObjective:
    lcfg = cfg["loss"]
    crit = TemporalObjective(
        w_l1=float(lcfg.get("l1", 1.0)),
        w_perceptual=float(lcfg.get("perceptual", 0.0)),
        net=str(lcfg.get("perceptual_net", "alex")),
        w_temporal=float(lcfg.get("temporal", 0.0)),
    ).to(device)
    if float(lcfg.get("perceptual", 0.0)) > 0.0 and not crit.used_perceptual:
        # Same raise as train_baseline: a ladder rung trained under a quietly
        # different objective is the confound KB Decision 35 exists to prevent.
        raise RuntimeError(
            "config requests a perceptual loss but the `lpips` package is unavailable; "
            "install the `train` extra rather than training under a different objective"
        )
    return crit


# --------------------------------------------------------------------------- #
# overfit — §11's mandatory local pre-cloud gate
# --------------------------------------------------------------------------- #
def overfit_frame_ids(train_ids: list[int], n: int, run_length: int) -> list[int]:
    """``n`` train frames as ``n // run_length`` evenly spread CONTIGUOUS runs.

    **A deviation from Model A's selection, forced by the data unit and recorded
    rather than hidden.** M4 picks ``n`` frames spread evenly across the training
    span, and evenly-spread frames are never adjacent — so they yield **zero**
    consecutive triples and a temporal model cannot be trained on them at all.
    Short runs, spread over the same span, are the smallest change that restores
    a trainable set while keeping the "not one contiguous block" property that
    made M4's choice non-vacuous.

    It is still a **weakening**: frames inside a run are near-duplicates at 60 fps,
    so 8 runs of 4 are easier to memorise than 32 scattered frames. No threshold
    is relaxed to compensate — M4's four thresholds are imported unchanged — and
    the direction of the weakening is stated here so it is not read as equivalent.
    """
    if n % run_length:
        raise ValueError(f"overfit n_frames {n} is not a multiple of run_length {run_length}")
    starts = np.linspace(0, len(train_ids) - run_length, n // run_length).round().astype(int)
    ids: list[int] = []
    for s in starts:
        ids.extend(train_ids[s:s + run_length])
    return sorted(set(ids))


def run_overfit(cfg: dict, device: torch.device) -> dict:
    o = cfg["overfit"]
    torch.manual_seed(o["seed"])
    np.random.seed(o["seed"])
    if device.type == "cuda":
        torch.cuda.empty_cache()

    session, target_dir, split = resolve_data(cfg)
    frame_ids = overfit_frame_ids(split.train_ids, int(o["n_frames"]), int(o["run_length"]))
    ds = TemporalTripleDataset(session, target_dir, frame_ids)
    criterion = build_criterion(cfg, device)

    model = build_model(cfg).to(device)
    init_note = None
    if cfg["train"].get("init_from"):
        init_note = init_from_checkpoint(model, ROOT / cfg["train"]["init_from"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]), weight_decay=0.0)

    loader = DataLoader(ds, batch_size=int(o["batch_size"]), shuffle=True,
                        num_workers=int(o.get("num_workers", 2)), drop_last=False,
                        persistent_workers=int(o.get("num_workers", 2)) > 0)
    model.train()
    max_steps, step0_loss, history, step = int(o["max_steps"]), None, [], 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            b = _to(batch, device)
            pred, pred_prev = training_forward(model, b)
            loss, _parts = criterion(pred, b["target_curr"], pred_prev,
                                     b["motion_curr"], b["mask_curr"])
            value = assert_finite_loss(loss, step=step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step0_loss is None:
                step0_loss = value
            if (step + 1) % 200 == 0:
                history.append({"step": step + 1, "loss": value})
            step += 1

    model.eval()
    l1s, psnrs, ssims = [], [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=2, shuffle=False):
            b = _to(batch, device)
            pred, _ = training_forward(model, b)
            pred, y = pred.float(), b["target_curr"]
            l1s.append((pred - y).abs().flatten(1).mean(dim=1).cpu())
            psnrs.append(psnr(pred, y).cpu())
            ssims.append(ssim(pred, y).cpu())
    final_l1 = float(torch.cat(l1s).mean())
    final_psnr = float(torch.cat(psnrs).mean())
    final_ssim = float(torch.cat(ssims).mean())
    if device.type == "cuda":
        torch.cuda.empty_cache()

    th = o["thresholds"]
    result = {
        "session_id": session.session_id,
        "style_id": cfg["data"]["style_id"],
        "arch": arch_of(cfg),
        "ladder_model": cfg.get("ladder_model"),
        "objective": criterion.describe(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "frame_ids": frame_ids,
        "n_frames": len(frame_ids),
        "n_triples": len(ds),
        "run_length": int(o["run_length"]),
        "frame_selection_note": overfit_frame_ids.__doc__.strip().splitlines()[0],
        "max_steps": max_steps,
        "batch_size": int(o["batch_size"]),
        "lr": float(o["lr"]),
        "seed": o["seed"],
        "init": init_note,
        "step0_loss": step0_loss,
        "objective_note": (
            "step0_loss / history[].loss are the TOTAL objective including the temporal "
            "term; final_l1 / psnr_db / ssim are measured quantities. loss_drop_factor is "
            "therefore objective-over-L1 — a MIXED-UNIT ratio, and a sanity check only "
            "(KB Decision 44)."
        ),
        "final_l1": final_l1,
        "loss_drop_factor": step0_loss / max(final_l1, 1e-12),
        "psnr_db": final_psnr,
        "ssim": final_ssim,
        "thresholds": th,
        "history": history,
    }
    result["checks"] = {
        "psnr_ge_min": final_psnr >= th["min_psnr_db"],
        "final_l1_le_max": final_l1 <= th["max_final_l1"],
        "loss_drop_ge_factor": result["loss_drop_factor"] >= th["min_loss_drop_factor"],
        "ssim_ge_min": final_ssim >= th["min_ssim"],
    }
    result["passed"] = all(result["checks"].values())
    return result


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def run_training(cfg: dict, device: torch.device, epochs: int | None = None,
                 quiet: bool = False) -> dict:
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    epochs = int(epochs if epochs is not None else t["epochs"])

    # Session-level split (KB Decision 64) when the config declares one; the
    # single-session path is untouched so every historical config still resolves
    # exactly as it did.
    if cfg["data"].get("styles"):
        # Two-style B: every (session, style), per-frame exclusion (KB Decision 94).
        from renderer.training.multi_session import (build_two_style_temporal_splits,
                                                     load_session_split, split_report)
        _sp = load_session_split(cfg["data"]["split_file"])
        _ex = json.loads((ROOT / cfg["data"]["frame_exclusions"]).read_text(encoding="utf-8"))
        train_ds, val_ds = build_two_style_temporal_splits(cfg["data"]["styles"], _sp, _ex)
        split_info = {**split_report(_sp), "styles": cfg["data"]["styles"],
                      "frame_exclusions": cfg["data"]["frame_exclusions"],
                      "n_frames_excluded": _ex["n_frames_excluded"]}
        session = None
    elif cfg["data"].get("split_file"):
        from renderer.training.multi_session import (build_temporal_splits,
                                                     load_session_split, split_report)
        _sp = load_session_split(cfg["data"]["split_file"])
        train_ds, val_ds = build_temporal_splits(cfg["data"]["style_id"], _sp)
        split_info = split_report(_sp)
        session = None
    else:
        session, target_dir, split = resolve_data(cfg)
        train_ds = TemporalTripleDataset(session, target_dir, split.train_ids)
        val_ds = TemporalFrameDataset(session, target_dir, split.val_ids)
        split_info = split.as_report()
    loader = DataLoader(
        train_ds,
        batch_size=int(t["batch_size"]),
        shuffle=True,
        num_workers=int(t["num_workers"]),
        drop_last=True,
        persistent_workers=int(t["num_workers"]) > 0,
    )

    model = build_model(cfg).to(device)
    init_note = None
    if t.get("init_from"):
        init_note = init_from_checkpoint(model, ROOT / t["init_from"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # T_max counts OPTIMISER steps, not batches — the correction KB Decision 47
    # records. Same construction as train_baseline so the two schedules are the
    # same shape; `warmup_epochs: 0` reproduces plain cosine exactly.
    accum = max(1, int(t.get("accum_steps", 1)))
    opt_steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = epochs * opt_steps_per_epoch
    warmup_steps = int(t.get("warmup_epochs", 0)) * opt_steps_per_epoch
    if warmup_steps > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, total_steps - warmup_steps)),
            ],
            milestones=[warmup_steps],
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    criterion = build_criterion(cfg, device)
    amp = _amp_dtype(t["amp_dtype"])
    use_amp = device.type == "cuda" and amp != torch.float32

    ckpt_dir = ROOT / t["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lp = LpipsMetric(device=device) if lpips_available() else None
    history, best = [], {"val_l1": float("inf"), "epoch": -1}
    started = time.time()
    # Early stopping, the SAME rule every other rung trains under (KB Decision 62:
    # patience counted in evaluations, a min_rel_delta so 1e-5 wobble does not
    # read as progress). This trainer lacked it until Decision 82, so the temporal
    # rungs ran to their epoch cap. It cannot change which checkpoint is selected
    # — best.pt is still the lowest val_l1 ever seen — only how long training runs.
    _es = t.get("early_stopping") or {}
    _patience = int(_es.get("patience_evals", 0))
    _min_rel = float(_es.get("min_rel_delta", 0.0))
    stale_evals, stopped_early, _best_for_es = 0, None, float("inf")

    for epoch in range(epochs):
        model.train()
        run_loss, run_temp, n = 0.0, 0.0, 0
        n_batches = len(loader)
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            b = _to(batch, device)
            with torch.autocast("cuda", dtype=amp, enabled=use_amp):
                pred, pred_prev = training_forward(model, b)
                loss, parts = criterion(pred, b["target_curr"], pred_prev,
                                        b["motion_curr"], b["mask_curr"])
            value = assert_finite_loss(loss, step=step, epoch=epoch)
            # Divide by the microbatches actually in this group, not by `accum`
            # (KB Decision 51's correction).
            group_start = (step // accum) * accum
            group_size = min(accum, n_batches - group_start)
            (loss / group_size).backward()
            if (step + 1) % accum == 0 or (step + 1) == n_batches:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            bs = b["structure_curr"].shape[0]
            run_loss += value * bs
            run_temp += parts.get("temporal", 0.0) * bs
            n += bs
        entry = {"epoch": epoch, "train_loss": run_loss / max(n, 1),
                 "train_temporal": run_temp / max(n, 1),
                 "train_objective": criterion.describe()}
        if (epoch + 1) % int(t["eval_every_epochs"]) == 0 or epoch == epochs - 1:
            val = evaluate(model, val_ds, device, lpips_metric=lp)
            entry.update({f"val_{k}": v for k, v in val.items()})
            if val["l1"] < best["val_l1"]:
                best = {"val_l1": val["l1"], "epoch": epoch,
                        **{f"val_{k}": v for k, v in val.items()}}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arch": arch_of(cfg),
                        "model_kwargs": model_kwargs(cfg),
                        "epoch": epoch,
                        "config_version": cfg["config_version"],
                        "split": split_info,
                        "val": val,
                    },
                    ckpt_dir / "best.pt",
                )
        if "val_l1" in entry:
            if entry["val_l1"] < _best_for_es * (1.0 - _min_rel):
                _best_for_es, stale_evals = entry["val_l1"], 0
            else:
                stale_evals += 1
        history.append(entry)
        if not quiet:
            print(json.dumps(entry), flush=True)
        if _patience and stale_evals >= _patience:
            stopped_early = {"stopped_at_epoch": epoch, "best_epoch": best["epoch"],
                             "patience_evals": _patience, "min_rel_delta": _min_rel,
                             "reason": (f"val_l1 did not improve by >{_min_rel:.2%} for {_patience} "
                                        f"consecutive evaluations; best was epoch {best['epoch']}")}
            if not quiet:
                print(json.dumps({"early_stop": stopped_early}), flush=True)
            break

    return {
        "epochs": epochs,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "steps_per_epoch": len(loader),
        "optimiser_steps_per_epoch": opt_steps_per_epoch,
        "samples_per_epoch": len(loader) * int(t["batch_size"]),
        "wall_seconds": time.time() - started,
        "parameters": parameter_count(model),
        "arch": arch_of(cfg),
        "ladder_model": cfg.get("ladder_model"),
        "split": split_info,
        "n_train_triples": len(train_ds),
        "n_val_frames": len(val_ds),
        "session_id": (session.session_id if session is not None else f"session_split:{cfg['data']['split_file']}"),
        "style_id": cfg["data"].get("style_id"),
        "styles": cfg["data"].get("styles"),
        "objective": criterion.describe(),
        "init": init_note,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "history": history,
        "best": best,
        "checkpoint": str(ckpt_dir / "best.pt"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Train ladder Model D (M6 temporal StructFormer).")
    ap.add_argument("--config", default=str(ROOT / "configs" / "temporal" / "m6_temporal.json"))
    ap.add_argument("--mode", choices=("train", "overfit"), default="train")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.report is None:
        assert_reports_resolvable(cfg, args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[m{cfg.get('milestone', 6)}] device={device} config={cfg['config_version']}", flush=True)

    if args.mode == "overfit":
        report = run_overfit(cfg, device)
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
        out = Path(args.report or report_path(cfg, "overfit"))
    else:
        report = run_training(cfg, device, epochs=args.epochs)
        out = Path(args.report or report_path(cfg, "train"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[m{cfg.get('milestone', 6)}] report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
