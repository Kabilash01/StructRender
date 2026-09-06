"""Milestone 4 — train / overfit / evaluate the CNN/U-Net baseline (ladder Model A).

    python -m renderer.training.train_baseline --mode train
    python -m renderer.training.train_baseline --mode overfit
    python -m renderer.training.train_baseline --mode eval --checkpoint <ckpt>

Frame-independent by construction: every frame is an independent sample, there is
no temporal state, no entity state and no EPAM (M6/M7/M8).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.image_quality import (
    LpipsMetric,
    gradient_energy,
    lpips_available,
    psnr,
    ssim,
)
from renderer.baselines.unet import UNetBaseline, parameter_count
from renderer.training.dataset import (
    Split,
    StyledFrameDataset,
    discover_session,
    make_split,
    target_dir_for,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "training" / "m4_unet_baseline.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def resolve_data(cfg: dict) -> tuple[StructuralSession, Path, Split]:
    d = cfg["data"]
    session_dir = d.get("session_dir") or discover_session()
    session = StructuralSession(session_dir)
    if d.get("require_validated") and not session.validated:
        raise RuntimeError(f"{session.session_id} has no VALIDATED marker")
    target_dir = target_dir_for(session.session_id, d["style_id"], d.get("target_root"))
    if not target_dir.is_dir():
        raise FileNotFoundError(f"no '{d['style_id']}' targets for {session.session_id} at {target_dir}")
    if d.get("require_aligned") and not (target_dir / "ALIGNED").is_file():
        raise RuntimeError(f"{target_dir} has no ALIGNED marker")
    split = make_split(session, cfg["split"]["val_sequences"], int(cfg["split"]["guard_frames"]))
    return session, target_dir, split


def overfit_frame_ids(split: Split, n: int) -> list[int]:
    """``n`` train frames spread evenly across the training span (fixed, seedless)."""
    idx = np.linspace(0, len(split.train_ids) - 1, n).round().astype(int)
    return [split.train_ids[i] for i in idx]


def high_camera_motion_frame_ids(
    session: StructuralSession, frame_ids: list[int], n: int, min_separation: int = 30
) -> list[int]:
    """The ``n`` frames of ``frame_ids`` with the largest frame-to-frame camera motion.

    The held-out split is two 8 s reappearance rungs, which hold the camera still
    for most of their length, so qualitative strips drawn only from it are
    near-static and low-information. Ranking by view-matrix delta finds the frames
    where the camera is actually moving. ``min_separation`` keeps the picks from
    collapsing onto ``n`` consecutive frames of one pan. Deterministic, no seed.
    """
    view = {
        fid: np.asarray(session.record(fid)["camera"]["view_matrix"], dtype=np.float64)
        for fid in frame_ids
    }
    ordered = sorted(view)
    delta = {
        b: float(np.abs(view[b] - view[a]).sum())
        for a, b in zip(ordered, ordered[1:])
        if b == a + 1
    }
    picked: list[int] = []
    for fid in sorted(delta, key=lambda f: (-delta[f], f)):
        if all(abs(fid - p) >= min_separation for p in picked):
            picked.append(fid)
        if len(picked) == n:
            break
    return sorted(picked)


# --------------------------------------------------------------------------- #
# numerical guard
# --------------------------------------------------------------------------- #
class NonFiniteLossError(RuntimeError):
    """A training step produced a NaN or Inf loss."""


def assert_finite_loss(loss: torch.Tensor, *, step: int, epoch: int | None = None) -> float:
    """KB §11 Local Pre-Cloud Training Gate, item 3 — "verify no NaN/Inf".

    Enforced inside the training loop rather than only after the fact: a run that
    goes non-finite at step 300 and keeps burning GPU hours is exactly what the
    gate exists to prevent, and the project's error-handling philosophy is
    explicit-over-silent with no auto-repair. Aborts loudly; never clamps,
    substitutes or skips the step.

    Returns the loss as a float so callers do not re-sync the device.
    """
    value = float(loss.detach())
    if not math.isfinite(value):
        where = f"step {step}" if epoch is None else f"epoch {epoch}, step {step}"
        raise NonFiniteLossError(
            f"non-finite training loss ({value}) at {where} — aborting "
            "(KB §11 pre-cloud gate item 3: verify no NaN/Inf)"
        )
    return value


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: StyledFrameDataset,
    device: torch.device,
    batch_size: int = 4,
    lpips_metric: LpipsMetric | None = None,
) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    l1s, psnrs, ssims, lps = [], [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).float()
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


@torch.no_grad()
def style_acquisition(
    model,
    dataset: StyledFrameDataset,
    device,
    batch_size: int = 4,
    lpips_metric: LpipsMetric | None = None,
) -> dict:
    """Exit criterion 3, quantified: is the output *styled*, or a passthrough — or blurred?

    Three families of number, all against the engine's own native render (used
    only as an evaluation reference — never a model input):

    - **closer than do-nothing.** ``*_pred_vs_target`` versus ``*_native_vs_target``
      for PSNR, SSIM and (when installed) LPIPS. The native anchors are what the
      "do nothing" baseline scores, so they bound how much range each metric
      actually has on this scene — measured, not assumed.
    - **moved off the native palette.** ``hist_pred_vs_native`` versus
      ``hist_target_vs_native``: colour-histogram L1 travel, the quantity M3's
      gate 4 uses (KB Decision 32, Finding 1) to reject a passthrough.
    - **did not get there by blurring.** ``sharpness_ratio`` = mean Sobel gradient
      energy of the prediction over that of the target. The histogram term alone
      cannot tell restyling from smoothing — a blurred frame keeps its shifted
      palette and so scores *higher*, not lower. M3's gate 4 was calibrated
      against a teacher whose failure mode was doing nothing; a trained
      regressor's failure mode is blur, which that metric rewards. A ratio well
      below 1 means the prediction is softer than its target.
    """
    from data_pipeline.teacher.metrics import stylisation_distance

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    p_t, n_t, s_pt, s_nt, l_pt, l_nt = [], [], [], [], [], []
    h_pn, h_tn, g_p, g_t, g_n = [], [], [], [], []
    for x, y, fids in loader:
        pred = model(x.to(device)).float().cpu()
        y = y.float()
        p_t.append(psnr(pred, y))
        s_pt.append(ssim(pred, y))
        g_p.append(gradient_energy(pred))
        g_t.append(gradient_energy(y))
        if lpips_metric is not None:
            l_pt.append(lpips_metric(pred, y).cpu())
        for i, fid in enumerate(fids.tolist()):
            native = dataset.session.load_native_rgb(fid)
            target = dataset.load_target(fid)
            pred_u8 = (pred[i].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            nat = torch.from_numpy(native.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            n_t.append(psnr(nat, y[i : i + 1]))
            s_nt.append(ssim(nat, y[i : i + 1]))
            g_n.append(gradient_energy(nat))
            if lpips_metric is not None:
                l_nt.append(lpips_metric(nat, y[i : i + 1]).cpu())
            h_pn.append(stylisation_distance(pred_u8, native)[1])
            h_tn.append(stylisation_distance(target, native)[1])
    out = {
        "psnr_pred_vs_target_db": float(torch.cat(p_t).mean()),
        "psnr_native_vs_target_db": float(torch.cat(n_t).mean()),
        "ssim_pred_vs_target": float(torch.cat(s_pt).mean()),
        "ssim_native_vs_target": float(torch.cat(s_nt).mean()),
        "hist_pred_vs_native": float(np.mean(h_pn)),
        "hist_target_vs_native": float(np.mean(h_tn)),
        "grad_energy_pred": float(torch.cat(g_p).mean()),
        "grad_energy_target": float(torch.cat(g_t).mean()),
        "grad_energy_native": float(torch.cat(g_n).mean()),
    }
    out["sharpness_ratio"] = out["grad_energy_pred"] / max(out["grad_energy_target"], 1e-12)
    out["sharpness_ratio_native"] = out["grad_energy_native"] / max(out["grad_energy_target"], 1e-12)
    if l_pt:
        out["lpips_pred_vs_target"] = float(torch.cat(l_pt).mean())
        out["lpips_native_vs_target"] = float(torch.cat(l_nt).mean())
    return out


@torch.no_grad()
def dump_qualitative(
    model,
    dataset: StyledFrameDataset,
    device,
    out_dir: Path,
    frame_ids: list[int],
    prefix: str = "frame",
) -> list[str]:
    """native | prediction | teacher target strips, for eyeballing criterion 3."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    written = []
    for fid in frame_ids:
        x, y, _ = dataset[dataset.frame_ids.index(fid)]
        pred = model(x[None].to(device)).float().cpu()[0]
        strip = np.concatenate(
            [
                dataset.session.load_native_rgb(fid),
                (pred.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8),
                dataset.load_target(fid),
            ],
            axis=1,
        )
        path = out_dir / f"{prefix}_{fid:05d}_native_pred_target.png"
        Image.fromarray(strip).save(path)
        written.append(str(path))
    return written


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _amp_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def run_overfit(cfg: dict, device: torch.device, log_every: int = 200) -> dict:
    """Exit criterion 2: overfit N frames. Thresholds live in the config."""
    o = cfg["overfit"]
    torch.manual_seed(o["seed"])
    np.random.seed(o["seed"])

    session, target_dir, split = resolve_data(cfg)
    frame_ids = overfit_frame_ids(split, int(o["n_frames"]))
    ds = StyledFrameDataset(session, target_dir, frame_ids)
    xs = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    ys = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)

    model = UNetBaseline(**_model_kwargs(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]), weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(o["seed"])

    model.train()
    bs, max_steps = int(o["batch_size"]), int(o["max_steps"])
    step0_loss = None
    history = []
    for step in range(max_steps):
        idx = torch.randint(0, len(ds), (bs,), generator=gen)
        loss = F.l1_loss(model(xs[idx]), ys[idx])
        value = assert_finite_loss(loss, step=step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step0_loss is None:
            step0_loss = value
        if (step + 1) % log_every == 0:
            history.append({"step": step + 1, "l1": value})

    model.eval()
    with torch.no_grad():
        pred = model(xs).float()
        final_l1 = float((pred - ys).abs().mean())
        final_psnr = float(psnr(pred, ys).mean())
        final_ssim = float(ssim(pred, ys).mean())

    th = o["thresholds"]
    result = {
        "frame_ids": frame_ids,
        "n_frames": len(frame_ids),
        "max_steps": max_steps,
        "batch_size": bs,
        "lr": float(o["lr"]),
        "seed": o["seed"],
        "step0_l1": step0_loss,
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
        # sanity check, not an independent gate — see overfit.threshold_notes
        "loss_drop_ge_factor": result["loss_drop_factor"] >= th["min_loss_drop_factor"],
        "ssim_ge_min": final_ssim >= th["min_ssim"],
    }
    result["passed"] = all(result["checks"].values())
    return result


def _model_kwargs(cfg: dict) -> dict:
    m = cfg["model"]
    return {
        "in_channels": m["in_channels"],
        "out_channels": m["out_channels"],
        "base_channels": m["base_channels"],
    }


def run_training(cfg: dict, device: torch.device, epochs: int | None = None, quiet: bool = False) -> dict:
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    epochs = int(epochs if epochs is not None else t["epochs"])

    session, target_dir, split = resolve_data(cfg)
    train_ds = StyledFrameDataset(session, target_dir, split.train_ids)
    val_ds = StyledFrameDataset(session, target_dir, split.val_ids)
    loader = DataLoader(
        train_ds,
        batch_size=int(t["batch_size"]),
        shuffle=True,
        num_workers=int(t["num_workers"]),
        drop_last=True,
        persistent_workers=int(t["num_workers"]) > 0,
    )

    model = UNetBaseline(**_model_kwargs(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    amp = _amp_dtype(t["amp_dtype"])
    use_amp = device.type == "cuda" and amp != torch.float32

    ckpt_dir = ROOT / t["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    lp = LpipsMetric(device=device) if lpips_available() else None
    history, best = [], {"val_l1": float("inf"), "epoch": -1}
    started = time.time()
    for epoch in range(epochs):
        model.train()
        run_loss, n = 0.0, 0
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp, enabled=use_amp):
                loss = F.l1_loss(model(x), y)
            value = assert_finite_loss(loss, step=step, epoch=epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += value * x.shape[0]
            n += x.shape[0]
        entry = {"epoch": epoch, "train_l1": run_loss / max(n, 1)}
        if (epoch + 1) % int(t["eval_every_epochs"]) == 0 or epoch == epochs - 1:
            val = evaluate(model, val_ds, device, lpips_metric=lp)
            entry.update({f"val_{k}": v for k, v in val.items()})
            if val["l1"] < best["val_l1"]:
                best = {"val_l1": val["l1"], "epoch": epoch, **{f"val_{k}": v for k, v in val.items()}}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_kwargs": _model_kwargs(cfg),
                        "epoch": epoch,
                        "config_version": cfg["config_version"],
                        "split": split.as_report(),
                        "val": val,
                    },
                    ckpt_dir / "best.pt",
                )
        history.append(entry)
        if not quiet:
            print(json.dumps(entry), flush=True)

    return {
        "epochs": epochs,
        "steps_per_epoch": len(loader),
        "wall_seconds": time.time() - started,
        "parameters": parameter_count(model),
        "split": split.as_report(),
        "session_id": session.session_id,
        "style_id": cfg["data"]["style_id"],
        "history": history,
        "best": best,
        "checkpoint": str(ckpt_dir / "best.pt"),
    }


def load_checkpoint(path: str | Path, device: torch.device) -> UNetBaseline:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNetBaseline(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M4 CNN/U-Net baseline (ladder Model A)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--mode", choices=("train", "overfit", "eval"), default="train")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[m4] device={device} config={cfg['config_version']}", flush=True)

    if args.mode == "overfit":
        report = run_overfit(cfg, device)
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
        out = Path(args.report or ROOT / "eval" / "reports" / "m4_overfit.json")
    elif args.mode == "train":
        report = run_training(cfg, device, epochs=args.epochs)
        out = Path(args.report or ROOT / "eval" / "reports" / "m4_train.json")
    else:
        session, target_dir, split = resolve_data(cfg)
        ckpt = args.checkpoint or ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
        model = load_checkpoint(ckpt, device)
        val_ds = StyledFrameDataset(session, target_dir, split.val_ids)
        train_ds = StyledFrameDataset(session, target_dir, split.train_ids)
        lp = LpipsMetric(device=device) if lpips_available() else None
        qual_dir = ROOT / "eval" / "reports" / "m4_qualitative"
        report = {
            "session_id": session.session_id,
            "style_id": cfg["data"]["style_id"],
            "checkpoint": str(ckpt),
            "parameters": parameter_count(model),
            "split": split.as_report(),
            "lpips_implementation": "lpips-alex-v0.1" if lp else "unavailable (lpips package not installed)",
            "selection_set_note": (
                "'selection_set' is the held-out split, but train_baseline.py checkpoints on its "
                "lowest val_l1, so these numbers are best-of-N-epochs on the same frames — the "
                "selection criterion's own minimum, not an unbiased generalisation estimate. "
                "Measured selection gain: best epoch 45 val_l1 0.004440 vs final epoch 59 0.004499, "
                "1.3% in L1 (~0.1 dB). KB Decision 35."
            ),
            "selection_set": evaluate(model, val_ds, device, lpips_metric=lp),
            "train_split_reference": evaluate(model, train_ds, device, lpips_metric=lp),
            "style_acquisition_selection_set": style_acquisition(model, val_ds, device, lpips_metric=lp),
            "qualitative": dump_qualitative(
                model, val_ds, device, qual_dir,
                [split.val_ids[i] for i in np.linspace(0, len(split.val_ids) - 1, 6).round().astype(int)],
            ),
            # The val split is two 8 s reappearance rungs and is deliberately near-static;
            # strips from it alone under-report what the model does under camera motion.
            "qualitative_train_moving_camera": dump_qualitative(
                model, train_ds, device, qual_dir,
                high_camera_motion_frame_ids(session, split.train_ids, 6),
                prefix="trainmove",
            ),
        }
        print(
            json.dumps(
                {k: v for k, v in report.items() if not k.startswith("qualitative")}, indent=2
            )
        )
        out = Path(args.report or ROOT / "eval" / "reports" / "m4_metrics.json")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[m4] report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
