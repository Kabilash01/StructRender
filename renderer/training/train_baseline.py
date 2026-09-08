"""Train / overfit / evaluate one frame-independent rung of the KB §11 ladder.

    python -m renderer.training.train_baseline --config configs/training/m4_unet_baseline.json --mode train
    python -m renderer.training.train_baseline --config configs/structformer/m5_structformer.json --mode train
    python -m renderer.training.train_baseline --mode overfit
    python -m renderer.training.train_baseline --mode eval --checkpoint <ckpt>

Both Model **A** (M4, CNN/U-Net) and Model **B** (M5, StructFormer) run through
THIS loop. That is deliberate and is the point of `build_model`: KB Decision 35
requires every ladder model to share the split, the selection rule and the
objective, and a forked training script is exactly how two rungs silently drift
apart on one of the three. The architecture is the only thing the config swaps.

Frame-independent by construction: every frame is an independent sample, there is
no temporal state, no entity state and no EPAM (M6/M7/M8).
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
from renderer.losses.perceptual import PerceptualReconstructionLoss
from renderer.structformer.model import StructFormerSingleStyle
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
def objective_string(cfg: dict) -> str:
    """Names the training objective for a report, from the config's numbers.

    Not the only implementation: the training and overfit loops write
    `PerceptualReconstructionLoss.describe()` instead, which formats the *live*
    criterion. The two are kept identical on purpose — reports derived from the
    config must match reports derived from the object that actually ran — and
    `test_reports_agree_on_the_objective` compares every committed report
    against this function, so a drift between them fails the suite rather than
    reaching an artifact.

    KB Decision 42 requires every committed report to name its objective. A
    clever one-liner here produced `"L_rec = L1 + 0"` — it split the config's
    prose note on "." and cut at the decimal point of `0.5*LPIPS` — so the
    metrics artifact claimed an L1-only run, which is the configuration
    Decision 41 records as FAILING the sharpness gate. A confidently wrong
    field is worse than a missing one: a missing field prompts a question, a
    wrong one closes it. Derived from the config's numbers, never from prose,
    and shared so the three reports cannot drift into three formats.
    """
    l = cfg["loss"]
    w = float(l.get("perceptual", 0.0))
    if w <= 0.0:
        return f"{float(l.get('l1', 1.0)):g}*L1"
    return f"{float(l.get('l1', 1.0)):g}*L1 + {w:g}*LPIPS({l.get('perceptual_net', 'alex')})"


#: Where each mode's report goes when a config does not say. These are M4's
#: historical names and exist ONLY for `m4_unet_baseline.json`, which predates the
#: `reports` block. Every later rung must declare its own paths, or two rungs
#: would overwrite one artifact.
LEGACY_REPORTS = {
    "train": "eval/reports/m4_train.json",
    "overfit": "eval/reports/m4_overfit.json",
    "metrics": "eval/reports/m4_metrics.json",
    "qualitative_dir": "eval/reports/m4_qualitative",
}


def report_path(cfg: dict, key: str) -> Path:
    """Absolute path for one of this config's reports."""
    return ROOT / cfg.get("reports", {}).get(key, LEGACY_REPORTS[key])


def _selection_note(cfg: dict) -> str:
    """Describe the checkpoint-selection bias using THIS run's own history.

    KB Decision 35 requires the bias to be stated inline with the metrics. It
    must be *derived*: a hardcoded figure published a retired scene's number into
    a new session's report, understating the real bias by ~2.5x (3.2 / 1.3).
    """
    base = ("'selection_set' is the held-out split, but train_baseline.py checkpoints on its "
            "lowest val_l1, so these numbers are best-of-N-epochs on the same frames — the "
            "selection criterion's own minimum, not an unbiased generalisation estimate. ")
    try:
        hist = json.loads(report_path(cfg, "train").read_text())
        vals = [(e["epoch"], e["val_l1"]) for e in hist["history"] if "val_l1" in e]
        if not vals:
            raise ValueError("no validated epochs")
        best_e, best_v = min(vals, key=lambda p: p[1])
        last_e, last_v = vals[-1]
        pct = 100.0 * (last_v - best_v) / best_v if best_v else float("nan")
        return base + (f"Measured selection gain on this run: best epoch {best_e} val_l1 "
                       f"{best_v:.6f} vs final epoch {last_e} {last_v:.6f}, {pct:.1f}% in L1. "
                       "KB Decision 35.")
    except Exception as exc:  # noqa: BLE001 — never block a report on the note
        return base + f"(selection gain unavailable: {exc}). KB Decision 35."


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

    # Release whatever earlier tests left resident. The overfit gate now trains
    # under the configured objective (KB Decision 42), which puts LPIPS's AlexNet
    # on the GPU as well as the model — enough extra that, on the 8 GB card, this
    # OOM'd inside the full suite while passing standalone. The gate must not be
    # decided by what ran before it.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    session, target_dir, split = resolve_data(cfg)
    frame_ids = overfit_frame_ids(split, int(o["n_frames"]))
    ds = StyledFrameDataset(session, target_dir, frame_ids)
    # Held on CPU and moved per batch. Pinning all N frames to the GPU cost
    # ~190 MB of headroom for no speed gain at this size.
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.stack([ds[i][1] for i in range(len(ds))])

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]), weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(o["seed"])

    # The overfit gate trains under the CONFIGURED objective, not a hardcoded
    # L1 (KB Decision 42). It is a capacity check on the setup we actually ship;
    # running it under a different loss than the ladder would tell us the model
    # can memorise something it is never asked to memorise.
    lcfg = cfg["loss"]
    criterion = PerceptualReconstructionLoss(
        w_l1=float(lcfg.get("l1", 1.0)),
        w_perceptual=float(lcfg.get("perceptual", 0.0)),
        net=str(lcfg.get("perceptual_net", "alex")),
    ).to(device)
    if float(lcfg.get("perceptual", 0.0)) > 0.0 and not criterion.used_perceptual:
        raise RuntimeError("config requests a perceptual loss but `lpips` is unavailable")

    model.train()
    bs, max_steps = int(o["batch_size"]), int(o["max_steps"])
    step0_loss = None
    history = []
    for step in range(max_steps):
        idx = torch.randint(0, len(ds), (bs,), generator=gen)
        xb, yb = xs[idx].to(device, non_blocking=True), ys[idx].to(device, non_blocking=True)
        loss, _parts = criterion(model(xb), yb)
        value = assert_finite_loss(loss, step=step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step0_loss is None:
            step0_loss = value
        if (step + 1) % log_every == 0:
            history.append({"step": step + 1, "loss": value})

    model.eval()
    # Chunked: forwarding all N frames in one batch was the actual OOM site
    # (a single 576 MiB activation on an 8 GB card already holding the model,
    # LPIPS and whatever the preceding tests left behind). Metrics are averaged
    # per frame, so chunking changes nothing but peak memory.
    with torch.no_grad():
        l1s, psnrs, ssims = [], [], []
        for i in range(0, len(ds), 4):
            xb = xs[i:i + 4].to(device)
            yb = ys[i:i + 4].to(device)
            pb = model(xb).float()
            l1s.append((pb - yb).abs().flatten(1).mean(dim=1).cpu())
            psnrs.append(psnr(pb, yb).cpu())
            ssims.append(ssim(pb, yb).cpu())
        final_l1 = float(torch.cat(l1s).mean())
        final_psnr = float(torch.cat(psnrs).mean())
        final_ssim = float(torch.cat(ssims).mean())
    if device.type == "cuda":
        torch.cuda.empty_cache()

    th = o["thresholds"]
    result = {
        # Self-identification (KB Decision 42): a report that does not name the
        # data and objective it came from cannot be told apart from a stale one.
        # An overfit artifact from a retired scene survived undetected precisely
        # because these four fields were absent.
        "session_id": session.session_id,
        "style_id": cfg["data"]["style_id"],
        "arch": cfg["model"].get("arch", "unet_baseline"),
        "objective": criterion.describe(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "frame_ids": frame_ids,
        "n_frames": len(frame_ids),
        "max_steps": max_steps,
        "batch_size": bs,
        "lr": float(o["lr"]),
        "seed": o["seed"],
        "step0_loss": step0_loss,
        "objective_note": (
            "step0_loss / history[].loss are the TOTAL objective (KB Decision 42); "
            "final_l1 / psnr_db / ssim are measured quantities. loss_drop_factor is "
            "therefore objective-over-L1 — a MIXED-UNIT ratio, and a sanity check only."
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
        # sanity check, not an independent gate — see overfit.threshold_notes
        "loss_drop_ge_factor": result["loss_drop_factor"] >= th["min_loss_drop_factor"],
        "ssim_ge_min": final_ssim >= th["min_ssim"],
    }
    result["passed"] = all(result["checks"].values())
    return result


#: ``arch`` -> (class, constructor keys read from ``cfg["model"]``). Adding a rung
#: means adding a row here, never a second training loop (KB Decision 35: the
#: ladder must measure the architecture, so split/selection/objective are shared).
ARCHITECTURES: dict[str, tuple[type, tuple[str, ...]]] = {
    "unet_baseline": (UNetBaseline, ("in_channels", "out_channels", "base_channels")),
    "structformer": (
        StructFormerSingleStyle,
        ("in_channels", "out_channels", "image_size", "embed_dim", "depths",
         "num_heads", "window_sizes", "style_dim", "attn_impl", "shift_impl"),
    ),
}


def _model_kwargs(cfg: dict) -> dict:
    """Constructor kwargs for ``cfg["model"]["arch"]``, ignoring its ``*_note`` prose.

    ``arch`` defaults to the U-Net so `m4_unet_baseline.json`, which predates this
    key, keeps working unchanged — and so an M4 checkpoint saved before the factory
    existed still loads.
    """
    m = cfg["model"]
    arch = m.get("arch", "unet_baseline")
    if arch not in ARCHITECTURES:
        raise KeyError(f"unknown model.arch {arch!r}; known: {sorted(ARCHITECTURES)}")
    _cls, keys = ARCHITECTURES[arch]
    kwargs = {k: m[k] for k in keys if k in m}
    for k in ("image_size", "depths", "num_heads", "window_sizes"):
        if k in kwargs:
            kwargs[k] = tuple(kwargs[k])
    return kwargs


def build_model(cfg: dict) -> torch.nn.Module:
    """Instantiate the configured ladder model. The ONLY construction site."""
    arch = cfg["model"].get("arch", "unet_baseline")
    return ARCHITECTURES[arch][0](**_model_kwargs(cfg))


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

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # T_max counts OPTIMISER steps, not batches. With gradient accumulation the
    # two differ, and sizing this by len(loader) would step cosine only 1/accum of
    # the way through its schedule — the LR would never anneal and Model C would
    # train under a different schedule than Model A while appearing to share one.
    _accum = max(1, int(t.get("accum_steps", 1)))
    _opt_steps_per_epoch = math.ceil(len(loader) / _accum)
    _total_steps = epochs * _opt_steps_per_epoch
    # Linear warmup then cosine. `warmup_epochs: 0` is the default and reproduces
    # the plain CosineAnnealingLR path exactly, so Model A's schedule is untouched
    # (KB Decision 48). Warmup exists for Model C's second attempt: the first ran
    # under Model A's schedule with none, which is the standard suspect for a soft
    # transformer, and it is the ONLY schedule change made.
    _warmup_steps = int(t.get("warmup_epochs", 0)) * _opt_steps_per_epoch
    if _warmup_steps > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-3, end_factor=1.0, total_iters=_warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, _total_steps - _warmup_steps)),
            ],
            milestones=[_warmup_steps],
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_total_steps)

    lcfg = cfg["loss"]
    criterion = PerceptualReconstructionLoss(
        w_l1=float(lcfg.get("l1", 1.0)),
        w_perceptual=float(lcfg.get("perceptual", 0.0)),
        net=str(lcfg.get("perceptual_net", "alex")),
    ).to(device)
    if float(lcfg.get("perceptual", 0.0)) > 0.0 and not criterion.used_perceptual:
        # Refuse to train a ladder model under a silently different objective —
        # KB Decision 35: the comparison must measure the architecture, not the loss.
        raise RuntimeError(
            "config requests a perceptual loss but the `lpips` package is unavailable; "
            "install the `train` extra rather than training under a different objective"
        )
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
        # Gradient accumulation exists for ONE reason (KB Decision 47): parameter
        # parity with Model A puts StructFormer at embed 240, which OOMs at batch
        # 8 on an 8 GB card. Running it at batch 4 instead would make batch size a
        # SECOND difference between two ladder rungs that are supposed to differ
        # only in architecture. accum_steps restores the effective batch, so the
        # optimiser sees the same 8 samples per update that Model A saw.
        # `accum_steps: 1` is the default and is exactly the old code path.
        accum = int(t.get("accum_steps", 1))
        n_batches = len(loader)
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp, enabled=use_amp):
                loss, _parts = criterion(model(x), y)
            value = assert_finite_loss(loss, step=step, epoch=epoch)
            # Divide by the microbatches actually IN THIS GROUP, not by `accum`.
            # The last group of an epoch can be short (161 batches at accum 2 gives
            # 80 full groups and one group of one), and dividing a short group by
            # `accum` applies a fractional-magnitude update — a half-weight step in
            # the M5 case. The committed M5 runs were trained with the old
            # `/accum` form; its effect on those numbers is UNMEASURED (KB
            # Decision 51). This is the corrected behaviour for future runs.
            group_start = (step // accum) * accum
            group_size = min(accum, n_batches - group_start)
            (loss / group_size).backward()
            if (step + 1) % accum == 0 or (step + 1) == n_batches:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += value * x.shape[0]
            n += x.shape[0]
        # NAME THE OBJECTIVE, NOT A METRIC. `run_loss` accumulates the total
        # training loss, which is L1 + w*LPIPS whenever a perceptual term is
        # configured (KB Decision 41). Publishing it as "train_l1" made the
        # report claim an L1 it never measured, and a ladder comparison against
        # M5's curve would have compared different quantities.
        entry = {"epoch": epoch, "train_loss": run_loss / max(n, 1),
                 "train_objective": criterion.describe()}
        if (epoch + 1) % int(t["eval_every_epochs"]) == 0 or epoch == epochs - 1:
            val = evaluate(model, val_ds, device, lpips_metric=lp)
            entry.update({f"val_{k}": v for k, v in val.items()})
            if val["l1"] < best["val_l1"]:
                best = {"val_l1": val["l1"], "epoch": epoch, **{f"val_{k}": v for k, v in val.items()}}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arch": cfg["model"].get("arch", "unet_baseline"),
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
        "arch": cfg["model"].get("arch", "unet_baseline"),
        "split": split.as_report(),
        "session_id": session.session_id,
        "style_id": cfg["data"]["style_id"],
        # KB Decision 42: every committed report names the data AND the objective
        # it was produced under. Without this a metrics file from an L1-only run
        # is indistinguishable from a perceptual one — which is the whole finding
        # of Decision 41.
        "objective": criterion.describe(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "history": history,
        "best": best,
        "checkpoint": str(ckpt_dir / "best.pt"),
    }


def load_checkpoint(path: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Checkpoints written before the factory existed carry no `arch`; they are all
    # Model A, so that is the default rather than an error.
    model = ARCHITECTURES[ckpt.get("arch", "unet_baseline")][0](**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train one frame-independent KB §11 ladder rung (Model A: M4 U-Net; "
                    "Model C: M5 StructFormer). The architecture comes from --config."
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--mode", choices=("train", "overfit", "eval"), default="train")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"m{cfg.get('milestone', 4)}"
    print(f"[{tag}] device={device} config={cfg['config_version']}", flush=True)

    if args.mode == "overfit":
        report = run_overfit(cfg, device)
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
        out = Path(args.report or report_path(cfg, "overfit"))
    elif args.mode == "train":
        report = run_training(cfg, device, epochs=args.epochs)
        out = Path(args.report or report_path(cfg, "train"))
    else:
        session, target_dir, split = resolve_data(cfg)
        ckpt = args.checkpoint or ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
        model = load_checkpoint(ckpt, device)
        val_ds = StyledFrameDataset(session, target_dir, split.val_ids)
        train_ds = StyledFrameDataset(session, target_dir, split.train_ids)
        lp = LpipsMetric(device=device) if lpips_available() else None
        qual_dir = report_path(cfg, "qualitative_dir")
        report = {
            "session_id": session.session_id,
            "style_id": cfg["data"]["style_id"],
            "checkpoint": str(ckpt),
            "arch": cfg["model"].get("arch", "unet_baseline"),
            "parameters": parameter_count(model),
            "split": split.as_report(),
            # KB Decision 42 self-identification. `objective` matters most here:
            # Decision 41's finding is that the objective drove the sharpness/SSIM
            # trade, so a metrics file without it cannot be distinguished from an
            # L1-only run.
            "objective": objective_string(cfg),
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "lpips_implementation": "lpips-alex-v0.1" if lp else "unavailable (lpips package not installed)",
            # Computed from THIS run's history, never quoted. A hardcoded figure
            # here published a retired scene's 1.3% selection gain into a new
            # session's report (KB Decision 42) — the third stale-citation defect
            # in this project, and the only durable fix is to derive it.
            "selection_set_note": _selection_note(cfg),
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
        out = Path(args.report or report_path(cfg, "metrics"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{tag}] report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
