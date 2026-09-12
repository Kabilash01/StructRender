"""Does cross-frame conditioning reduce the teacher's 35x temporal instability?

KB Decision 55/57 measured the `oil_impasto` targets at ~35x the engine's own
temporal error on the frames that move. Decision 57 identified that as M6's root
cause: a temporal rung trained on per-frame-independent supervision is taught to
reproduce flicker, and its temporal loss is outvoted by reconstruction.

**This is a PROBE, not a regeneration.** Regenerating 990 targets and retraining
the whole ladder is roughly a day of compute; this answers whether that would pay
before spending it, the same sequencing that made the M5 and M6 export probes
worth their cost.

The change under test: instead of img2img-ing from the NATIVE frame every time,
seed frame t from the previous STYLED frame warped into t by the engine's own
motion field, and fall back to native where M6's confidence mask says the warp is
untrustworthy (disocclusion, out of bounds, background):

    init_t = M_t * W(styled_{t-1}, V_t) + (1 - M_t) * native_t

Nothing about the seed, prompt, strength, guidance or ControlNet scale changes.
Writes ``eval/reports/m3_temporal_teacher_probe.json``.

    python -m scripts.probe_m3_temporal_teacher      (needs CUDA + diffusers)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from data_pipeline.teacher.styles import load_style
from eval.metrics.temporal import engine_confidence
from renderer.temporal.pairs import consecutive_pairs
from renderer.temporal.warp import EngineGuidedWarp
from renderer.training.dataset import discover_session
from renderer.training.train_baseline import ROOT


def _to_t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]


def _temporal_error(seq: dict, pairs, session, warp, device) -> float:
    """Mask-weighted |I_t - W(I_{t-1})|, the same quantity the M6 gate uses.

    Everything is moved to one device explicitly: `engine_confidence` returns CPU
    tensors and the warp lives on the GPU.
    """
    errs = []
    for p in pairs:
        if p.prev_id not in seq or p.curr_id not in seq:
            continue
        c = engine_confidence(session, p)
        prev = _to_t(seq[p.prev_id]).to(device)
        curr = _to_t(seq[p.curr_id]).to(device)
        warped = warp(prev, c["motion"].to(device))
        m = c["mask"].to(device)
        d = (curr - warped).abs().mean(dim=1, keepdim=True)
        errs.append(float((d * m).sum() / m.sum().clamp_min(1.0)))
    return float(np.mean(errs)) if errs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 cross-frame teacher conditioning probe")
    ap.add_argument("--style", default="oil_impasto")
    ap.add_argument("--span", type=int, nargs=2, default=[439, 468],
                    help="a MOVING span; the default is inside the 4s rung, guard band")
    ap.add_argument("--out", default="eval/reports/m3_temporal_teacher_probe.json")
    args = ap.parse_args()

    session = StructuralSession(discover_session())
    style = load_style(args.style)
    device = torch.device("cuda")
    warp = EngineGuidedWarp(288, 512).to(device).eval()

    lo, hi = args.span
    frames = [f for f in range(lo, hi + 1) if f in set(session.frame_ids())]
    pairs = [p for p in consecutive_pairs(session, frames) if p.tick_delta == 1]
    print(f"[probe] span {lo}-{hi}: {len(frames)} frames, {len(pairs)} dt==1 pairs", flush=True)

    from data_pipeline.teacher.adapter import make_backend
    backend = make_backend(style)

    # --- A: current behaviour — img2img from the native frame, every frame -----
    baseline: dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        native = session.load_native_rgb(f)
        baseline[f] = backend.generate(native, session.load_structure(f), style).styled_rgb
        print(f"  [A independent] {i+1}/{len(frames)}", end="\r", flush=True)
    print()

    # --- B: proposed — seed from the warped previous STYLED frame --------------
    chained: dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        native = session.load_native_rgb(f)
        prev = f - 1
        if prev in chained and session.motion_valid(f):
            c = engine_confidence(session, next(p for p in pairs if p.curr_id == f)) \
                if any(p.curr_id == f for p in pairs) else None
            if c is not None:
                w = warp(_to_t(chained[prev]).to(device), c["motion"].to(device))
                m = c["mask"].to(device)
                blended = (m * w + (1.0 - m) * _to_t(native).to(device)).clamp(0, 1)
                init = (blended[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                init = native
        else:
            init = native  # cold start: identical to A
        chained[f] = backend.generate(init, session.load_structure(f), style).styled_rgb
        print(f"  [B cross-frame] {i+1}/{len(frames)}", end="\r", flush=True)
    print()

    native_seq = {f: session.load_native_rgb(f) for f in frames}
    te_native = _temporal_error(native_seq, pairs, session, warp, device)
    te_a = _temporal_error(baseline, pairs, session, warp, device)
    te_b = _temporal_error(chained, pairs, session, warp, device)

    out = {
        "purpose": ("Probe: does cross-frame conditioning reduce the teacher's temporal "
                    "instability, which KB Decision 57 identified as M6's root cause?"),
        "not_a_regeneration": ("This generates one short span, not the 990-frame target set. It "
                               "answers whether a regeneration would pay before it is spent."),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "session_id": session.session_id, "style_id": args.style,
        "span": [lo, hi], "n_frames": len(frames), "n_pairs": len(pairs),
        "metric": "mask-weighted mean |I_t - W(I_{t-1}, V_t)| — the M6 gate's quantity",
        "change_under_test": ("init_t = M_t * W(styled_{t-1}, V_t) + (1 - M_t) * native_t, "
                              "instead of init_t = native_t. Seed, prompt, strength, guidance and "
                              "controlnet scale are unchanged."),
        "temporal_error": {
            "native_engine": te_native,
            "teacher_independent_frames": te_a,
            "teacher_cross_frame": te_b,
        },
        "ratios": {
            "independent_vs_engine": te_a / te_native if te_native else None,
            "cross_frame_vs_engine": te_b / te_native if te_native else None,
            "improvement_pct": 100 * (te_b - te_a) / te_a if te_a else None,
        },
        "caveats": ("Cross-frame conditioning makes generation SEQUENTIAL (no parallelism) and can "
                    "accumulate drift over a long span; this probe covers a short one. It also "
                    "changes what the alignment gates see, so KB Decision 32's four gates would "
                    "have to be re-run on any regenerated set."),
    }
    p = ROOT / args.out
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[probe] wrote {p}")
    print(f"  native engine            {te_native:.6f}")
    print(f"  teacher, independent     {te_a:.6f}   ({te_a/te_native:.1f}x engine)")
    print(f"  teacher, cross-frame     {te_b:.6f}   ({te_b/te_native:.1f}x engine)")
    print(f"  change: {100*(te_b-te_a)/te_a:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
