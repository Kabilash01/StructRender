"""M6's temporal-error baseline, measured BEFORE Model D exists.

KB §10 Milestone 6's exit criterion is "lower temporal error than
frame-independent StructFormer". A criterion stated as a comparison needs the
other side of it measured first, and it needs a *spread* — otherwise "lower"
means "lower by any amount", including by less than one run's noise. This runner
produces both:

* **TE for every existing source** — the native engine, the teacher targets, all
  three Model A (CNN) seeds and both Model C (StructFormer) attempts;
* **the across-seed spread of the three CNN seeds**, which is the run-to-run
  variance the M6 margin is to be pre-registered against. Exactly the
  construction KB Decision 46 used for M5's margin, applied to a new metric.

**No training happens here.** The checkpoints are the committed ones.

Writes ``eval/reports/m6_temporal_baseline.json``.

    python -m eval.protocols.run_temporal_baseline
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.temporal import (
    KEEP_THRESHOLD,
    MIN_KEPT_PIXELS,
    engine_confidence,
    evaluate_source,
    summarise,
)
from renderer.temporal.confidence import ALPHA, BETA, E_D_HALF_METRES, E_N_HALF
from renderer.temporal.pairs import consecutive_pairs
from renderer.training.dataset import build_input, discover_session, make_split, target_dir_for
from renderer.training.train_baseline import ROOT, load_checkpoint, load_config

#: Every trained rung that exists. The three CNN seeds are what the margin is
#: built from, so they are not optional rows.
MODELS = {
    "m4_unet_baseline": "configs/training/m4_unet_baseline.json",
    "m4_unet_seed_repeat": "configs/training/m4_unet_seed_repeat.json",
    "m4_unet_seed3": "configs/training/m4_unet_seed3.json",
    "m5_structformer_attempt1": "configs/structformer/m5_structformer.json",
    "m5_structformer_attempt2_warmup": "configs/structformer/m5_structformer_warmup.json",
}
CNN_SEEDS = ("m4_unet_baseline", "m4_unet_seed_repeat", "m4_unet_seed3")

REQUIRED_TOP_LEVEL = (
    "milestone", "metric", "purpose", "generated_utc", "torch_version", "session_id",
    "style_id", "objective", "split", "mask", "gating", "sources",
    "cnn_across_seed_spread", "sharpness_caveat", "no_training_note",
    "motion_response_diagnostic", "measurability_warning",
)


def _target_loader(target_dir: Path):
    index = {int(r["frame_id"]): r for r in
             (json.loads(l) for l in
              (target_dir / "manifest.jsonl").read_text().splitlines() if l.strip())}

    def load(fid: int) -> np.ndarray:
        return np.asarray(Image.open(target_dir / index[fid]["target_rgb_path"]).convert("RGB"),
                          dtype=np.float32) / 255.0
    return load


def _native_loader(session):
    def load(fid: int) -> np.ndarray:
        return session.load_native_rgb(fid).astype(np.float32) / 255.0
    return load


def _model_loader(model, session, device):
    cache: dict[int, np.ndarray] = {}

    def load(fid: int) -> np.ndarray:
        if fid not in cache:
            x = torch.from_numpy(build_input(session.load_structure(fid)))[None].to(device)
            with torch.no_grad():
                y = model(x)[0].float().clamp(0, 1).cpu().numpy()
            cache[fid] = np.transpose(y, (1, 2, 0))
        return cache[fid]
    return load


def _stats_list(vals) -> dict:
    import statistics as _st
    v = [float(x) for x in vals]
    return {"n": len(v), "mean": _st.mean(v) if v else float("nan"),
            "sd": _st.stdev(v) if len(v) > 1 else None}


def _spread(values: dict) -> dict:
    a = np.asarray(list(values.values()), dtype=np.float64)
    return {"per_seed": values, "n": int(a.size),
            "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": float(a.min()), "max": float(a.max()),
            "range": float(a.max() - a.min())}


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 temporal-error baseline (KB §11 Temporal)")
    ap.add_argument("--config", default="configs/training/m4_unet_baseline.json")
    ap.add_argument("--out", default="eval/reports/m6_temporal_baseline.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = StructuralSession(cfg["data"].get("session_dir") or discover_session())
    target_dir = target_dir_for(session.session_id, cfg["data"]["style_id"],
                                cfg["data"].get("target_root"))
    split = make_split(session, cfg["split"]["val_sequences"], int(cfg["split"]["guard_frames"]))
    pairs = consecutive_pairs(session, split.val_ids)
    dt1 = sum(1 for p in pairs if p.tick_delta == 1)

    # The GUARD BAND is the M6 gate (KB Decision 55). `guard_frames = 60` drops
    # these frames from BOTH splits, so no model trained on them and no
    # checkpoint was selected on them — they are genuinely held out — and unlike
    # the validation rungs they contain real camera and entity motion, so
    # temporal error there measures reprojection rather than flicker.
    guard_ids = sorted(set(session.frame_ids()) - set(split.train_ids) - set(split.val_ids))
    guard_pairs = consecutive_pairs(session, guard_ids)
    guard_dt1 = sum(1 for p in guard_pairs if p.tick_delta == 1)

    print(f"[te] {len(pairs)} val pairs — {dt1} at dt==1, {len(pairs)-dt1} at dt>1", flush=True)
    print("[te] building the confidence masks once, shared by every source ...", flush=True)
    masks = {p.curr_id: engine_confidence(session, p) for p in pairs}
    print(f"[te] guard band: {len(guard_pairs)} pairs — {guard_dt1} at dt==1", flush=True)
    guard_masks = {p.curr_id: engine_confidence(session, p) for p in guard_pairs}

    target_load = _target_loader(target_dir)
    report = {
        "milestone": 6,
        "metric": "TE = mean |I_t - W(I_{t-1}, V_t)| over pixels with M_t > "
                  f"{KEEP_THRESHOLD} — KB §11 Core Metrics / Temporal",
        "purpose": ("Measure the temporal-error baseline for every EXISTING rung before Model D "
                    "is trained, so M6's margin can be pre-registered against the CNN "
                    "across-seed spread rather than chosen after seeing the result "
                    "(KB Decision 46's construction, applied to a new metric)."),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "session_id": session.session_id,
        "style_id": cfg["data"]["style_id"],
        "objective": "1*L1 + 0.5*LPIPS(alex)  — the objective every ladder rung trained under "
                     "(KB Decision 41). Stated because a TE comparison across rungs is only "
                     "about architecture if the objective is held fixed.",
        "split": {
            "val_sequences": split.val_sequences,
            "guard_frames": split.guard_frames,
            "n_val_frames": len(split.val_ids),
            "n_pairs": len(pairs), "n_pairs_dt_eq_1": dt1, "n_pairs_dt_gt_1": len(pairs) - dt1,
            "pairing_rule": "(f-1, f) with BOTH ends inside the validation split, so the two "
                            "sequence-start frames contribute no pair: 166 val frames -> 164 "
                            "pairs, not 166.",
            "selection_set_note": ("This split is also the checkpoint-selection set for every "
                                   "model scored here (KB Decision 35), so these are "
                                   "selection-set figures, not a clean generalisation estimate. "
                                   "TE itself was never a selection criterion — checkpoints were "
                                   "chosen on val_l1 — so it is not directly optimised, but it "
                                   "is measured on frames the selection touched."),
        },
        "mask": {
            "alpha": ALPHA, "beta": BETA,
            "e_d_half_metres": E_D_HALF_METRES, "e_n_half": E_N_HALF,
            "keep_threshold": KEEP_THRESHOLD,
            "calibration_artifact": "eval/reports/m6_mask_calibration.json",
            "shared": ("One mask per pair, computed from ENGINE buffers only and shared by every "
                       "source, so two sources cannot be scored on different pixels."),
        },
        "gating": ("dt_eq_1 is the headline gate (training is 100% dt==1). dt_gt_1 is reported "
                   "separately as out-of-distribution robustness: those pairs were recorded at a "
                   "reduced rate during the long reappearance gaps and carry ~8x the "
                   "displacement. Tick deltas are computed from each record's own 'tick' field."),
        "sharpness_caveat": (
            "TE MUST NOT BE READ WITHOUT THE SHARPNESS COLUMN. A blurrier renderer has trivially "
            "lower temporal error — blur removes the high-frequency content that sub-pixel "
            "misalignment turns into error. This is KB Decision 53's RCS confound on a new "
            "metric, and it is not hypothetical here: Model C already sits below M4's 0.90 "
            "sharpness floor, so the rung most likely to post a low TE is the one already known "
            "to be too soft."),
        "measurability_warning": (
            "READ THIS BEFORE USING dt_eq_1 AS A GATE. 43 of the 44 dt==1 validation pairs "
            "displace by under HALF A PIXEL (median 0.26 px; only one pair exceeds 1 px), and 63 "
            "of the 120 dt>1 pairs are exactly static. The held-out sequences are the two "
            "8-second reappearance rungs, which park the camera for most of their length — KB "
            "Decision 35 recorded that property for the qualitative strips and it lands here "
            "too. Consequence: on this split TE is very nearly the plain frame difference "
            "|I_t - I_{t-1}|, and warp_gain_vs_unwarped is ~0.95 for every source, i.e. warping "
            "makes it slightly WORSE because bilinear resampling costs more than a quarter-pixel "
            "of alignment buys. TE here still measures temporal STABILITY (flicker), which is a "
            "real property and is what separates the models from their teacher — but it does NOT "
            "measure reprojection quality, so M6's exit criterion as stated would be won by "
            "whichever model flickers least. See by_motion_bucket and "
            "motion_response_diagnostic. This is a property of the SPLIT, not of the warp: the "
            "warp is verified correct on frames that do move "
            "(renderer/temporal/tests/test_warp.py, median 2.04x error reduction)."),
        "no_training_note": "No model was trained, fine-tuned or re-selected by this run. Every "
                            "checkpoint is the committed one.",
        "sources": {},
    }

    print("[te] native_engine ...", flush=True)
    report["sources"]["native_engine"] = summarise(
        evaluate_source(session, _native_loader(session), pairs, target_load, masks))
    report["sources"]["native_engine"]["role"] = (
        "The engine's own render. Deterministic, so this is the temporal-error floor that the "
        "warp plus the scene's own disocclusion allows — not zero.")

    print("[te] teacher_target ...", flush=True)
    report["sources"]["teacher_target"] = summarise(
        evaluate_source(session, target_load, pairs, target_load, masks))
    report["sources"]["teacher_target"]["role"] = (
        "The supervision. A frame-independent student trained on per-frame targets cannot be "
        "more temporally consistent than these without a temporal mechanism — which is what M6 "
        "adds. Its sharpness ratio is 1.0 by construction.")

    for name, cfg_path in MODELS.items():
        mcfg = load_config(ROOT / cfg_path)
        ck = ROOT / mcfg["train"]["checkpoint_dir"] / "best.pt"
        if not ck.is_file():
            print(f"[te] {name}: no checkpoint at {ck} — SKIPPED", flush=True)
            report["sources"][name] = {"skipped": f"no checkpoint at {ck}"}
            continue
        print(f"[te] {name} ...", flush=True)
        model = load_checkpoint(ck, device).eval()
        s = summarise(evaluate_source(session, _model_loader(model, session, device),
                                      pairs, target_load, masks))
        s["checkpoint"] = str(ck.relative_to(ROOT))
        s["ladder_model"] = mcfg.get("ladder_model")
        s["train_seed"] = mcfg["train"]["seed"]
        report["sources"][name] = s
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---- does TE respond to warp quality at all on this session? -------------
    # It has to be asked, because 43 of the 44 headline dt==1 validation pairs
    # displace by under half a pixel: the held-out sequences are the two 8-second
    # reappearance rungs, which park the camera. On that split TE is very nearly
    # |I_t - I_{t-1}| under another name, and a warp cannot beat the identity at
    # a quarter-pixel offset.
    #
    # Run over EVERY pair in the session, not just the validation split, and
    # scored for the native engine and the teacher targets ONLY. Neither of those
    # was fit to anything, so which split a frame belongs to is irrelevant to
    # them — and the high-motion frames turn out to sit mostly in the guard band,
    # which belongs to no split at all. The trained models are deliberately
    # excluded: for them split membership would matter, and reporting a model on
    # its own training frames is not a measurement.
    # ---- the M6 gate: guard band, real motion, never trained or selected on ----
    print("[te] guard-band gate (all sources) ...", flush=True)
    gb: dict = {}
    gb["native_engine"] = summarise(
        evaluate_source(session, _native_loader(session), guard_pairs, target_load, guard_masks))
    gb["teacher_target"] = summarise(
        evaluate_source(session, target_load, guard_pairs, target_load, guard_masks))
    for name, cfg_path in MODELS.items():
        mcfg = load_config(ROOT / cfg_path)
        ck = ROOT / mcfg["train"]["checkpoint_dir"] / "best.pt"
        if not ck.is_file():
            gb[name] = {"skipped": f"no checkpoint at {ck}"}
            continue
        model = load_checkpoint(ck, device).eval()
        g = summarise(evaluate_source(session, _model_loader(model, session, device),
                                      guard_pairs, target_load, guard_masks))
        g["checkpoint"] = str(ck.relative_to(ROOT))
        g["ladder_model"] = mcfg.get("ladder_model")
        gb[name] = g
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    present_g = [k for k in CNN_SEEDS if "dt_eq_1" in gb.get(k, {})]
    report["guard_band_gate"] = {
        "role": ("THE M6 EXIT GATE (KB Decision 55). The validation rungs park the camera, so TE "
                 "there measures flicker, not reprojection. These 3 contiguous ranges are dropped "
                 "from both splits by guard_frames=60, so no model saw them in training or "
                 "selection, and they carry the 4-second rungs' real motion."),
        "frame_ranges": [[432, 491], [575, 634], [847, 906]],
        "n_frames": len(guard_ids), "n_pairs": len(guard_pairs), "n_pairs_dt_eq_1": guard_dt1,
        "held_out_note": ("Disjoint from train and val by construction; asserted in "
                          "eval/metrics/tests/test_temporal.py."),
        "sources": gb,
        "cnn_across_seed_spread": {
            "seeds": present_g,
            "note": ("The spread M6's margin is pre-registered from. It is a CNN quantity measured "
                     "over 3 seeds pooling a toolchain change (KB Decision 51): a preliminary "
                     "estimate, not a confidence interval, and NOT a measurement of StructFormer "
                     "variance, which remains unmeasured."),
            "dt_eq_1_temporal_error": _spread(
                {k: gb[k]["dt_eq_1"]["temporal_error"]["mean"] for k in present_g}),
        } if present_g else None,
    }

    # ---- the pre-registered M6 gate: guard band AND real motion ---------------
    # Threshold chosen on physics, not on the numbers: below ~1 px the bilinear
    # resampling error exceeds what alignment buys, which is why warp_gain is
    # BELOW 1 on sub-pixel pairs even for the deterministic engine. 1 px is the
    # smallest displacement at which a warp can pay for itself, and it maximises
    # the sample. Fixed here before Model D exists (KB Decision 55).
    MOTION_PX = 1.0
    _base = {q["curr_id"]: q.get("median_motion_px") or 0.0
             for q in gb["native_engine"]["per_pair"]}
    moving = {k for k, v in _base.items() if v >= MOTION_PX}
    gate_rows: dict = {}
    for name, sv in gb.items():
        if "per_pair" not in sv:
            continue
        rows = [q for q in sv["per_pair"] if q["curr_id"] in moving]
        if not rows:
            continue
        te = [q["te"] for q in rows]
        gate_rows[name] = {
            "n_pairs": len(rows),
            "temporal_error": _stats_list(te),
            "unwarped_anchor": _stats_list([q["unwarped"] for q in rows]),
            "warp_gain_vs_unwarped": (sum(q["unwarped"] for q in rows) / sum(te)) if sum(te) else None,
            "sharpness_ratio_vs_target": _stats_list(
                [q["grad_energy"] / q["grad_energy_target"] for q in rows
                 if q.get("grad_energy_target")]),
        }
    seeds_present = [k for k in CNN_SEEDS if k in gate_rows]
    spread = _spread({k: gate_rows[k]["temporal_error"]["mean"] for k in seeds_present}) \
        if len(seeds_present) == 3 else None
    model_c = gate_rows.get("m5_structformer_attempt1", {}).get("temporal_error", {}).get("mean")
    threshold = None
    if spread and model_c:
        rel = spread["sd"] / spread["mean"]
        threshold = {
            "model_c_temporal_error": model_c,
            "cnn_relative_sd": rel,
            "margin": 2 * rel,
            "model_d_must_be_at_most": model_c * (1 - 2 * rel),
            "rule": ("M6 passes if Model D's temporal error on this gate is at least 2 CNN "
                     "across-seed sd below Model C's, AND its sharpness does not fall more than "
                     "1 sd below Model C's. Fixed before Model D is trained (KB Decision 55)."),
            "sharpness_guard": {
                "model_c": gate_rows["m5_structformer_attempt1"]["sharpness_ratio_vs_target"]["mean"],
                "cnn_sd": _spread({k: gate_rows[k]["sharpness_ratio_vs_target"]["mean"]
                                   for k in seeds_present})["sd"],
                "why": ("A blurrier renderer has trivially lower temporal error (KB Decision 53's "
                        "RCS confound on a new metric). M6 must not win by softening."),
            },
        }
    report["m6_gate"] = {
        "role": "THE PRE-REGISTERED M6 EXIT GATE (KB Decision 55).",
        "definition": (f"Guard-band pairs whose median displacement is >= {MOTION_PX} px. The guard "
                       "band is held out of BOTH splits, and the motion filter is what makes "
                       "temporal error measure REPROJECTION rather than flicker."),
        "motion_threshold_px": MOTION_PX,
        "n_pairs": len(moving),
        "sources": gate_rows,
        "cnn_across_seed_spread": spread,
        "pre_registered_threshold": threshold,
    }

    print("[te] motion-response diagnostic (all pairs, >= 2 px) ...", flush=True)
    all_pairs = consecutive_pairs(session, session.frame_ids())
    diag_masks, diag_pairs = {}, []
    for p in all_pairs:
        m = engine_confidence(session, p)
        mo = m["motion"]
        mag = torch.sqrt((mo[:, :1] * mo.shape[-1]) ** 2 + (mo[:, 1:] * mo.shape[-2]) ** 2)
        keep = m["mask"] > KEEP_THRESHOLD
        if float(keep.sum()) < 5000:
            continue
        if float(mag[keep].median()) >= 2.0:
            diag_masks[p.curr_id] = m
            diag_pairs.append(p)
    report["motion_response_diagnostic"] = {
        "purpose": ("Whether TE measures warp quality AT ALL on this session, asked separately "
                    "because the validation split cannot answer it: 43 of its 44 dt==1 pairs "
                    "displace by under half a pixel."),
        "scope": ("Every pair in the session with >= 2 px median displacement, from any split. "
                  "NOT A GATE and not part of any model comparison: it scores the native engine "
                  "and the teacher targets ONLY, neither of which was fit to anything, so split "
                  "membership does not apply to them. Trained models are excluded on purpose. "
                  "Most qualifying frames fall in the 60-frame guard band, which belongs to no "
                  "split — another reason the validation split cannot answer this question."),
        "n_pairs": len(diag_pairs),
        "sources": {
            "native_engine": summarise(evaluate_source(
                session, _native_loader(session), diag_pairs, target_load, diag_masks)),
            "teacher_target": summarise(evaluate_source(
                session, target_load, diag_pairs, target_load, diag_masks)),
        } if diag_pairs else {},
    }

    # Pairs whose mask keeps too little to be a measurement are omitted by
    # eval.metrics.temporal, never imputed. Say how many, and where: a reader
    # comparing n_pairs against the split counts should not have to infer it.
    for name, s in report["sources"].items():
        if "dt_eq_1" not in s:
            continue
        s["pairs_omitted_too_few_kept_pixels"] = {
            "dt_eq_1": dt1 - s["dt_eq_1"]["n_pairs"],
            "dt_gt_1": (len(pairs) - dt1) - s["dt_gt_1"]["n_pairs"],
            "rule": f"a pair is dropped when the confidence mask keeps < "
                    f"{MIN_KEPT_PIXELS} pixels; dropped pairs are omitted, never imputed",
        }

    present = [k for k in CNN_SEEDS if "dt_eq_1" in report["sources"].get(k, {})]
    if len(present) >= 2:
        report["cnn_across_seed_spread"] = {
            "seeds": {k: report["sources"][k]["train_seed"] for k in present},
            "note": ("The number M6's margin is to be built from. Three Model A runs differing "
                     "ONLY in seed, so this is run-to-run variance and nothing else. A Model D "
                     "improvement smaller than this is not evidence (KB Decision 46)."),
            "dt_eq_1_temporal_error": _spread(
                {k: report["sources"][k]["dt_eq_1"]["temporal_error"]["mean"] for k in present}),
            "dt_gt_1_temporal_error": _spread(
                {k: report["sources"][k]["dt_gt_1"]["temporal_error"]["mean"] for k in present}),
            "dt_eq_1_sharpness_ratio": _spread(
                {k: report["sources"][k]["dt_eq_1"]["sharpness_ratio_vs_target"]["mean"]
                 for k in present}),
        }
    else:
        report["cnn_across_seed_spread"] = {
            "unavailable": f"only {len(present)} CNN seed checkpoints present: {present}"}

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in report]
    if missing:
        raise SystemExit(f"generator/schema drift — REQUIRED_TOP_LEVEL not produced: {missing}")

    out = Path(ROOT) / args.out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[te] wrote {out}\n")
    hdr = f"{'source':34s} {'TE dt=1':>10s} {'anchor':>9s} {'gain':>6s} {'sharp':>7s} | {'TE dt>1':>10s} {'sharp':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in report["sources"].items():
        if "dt_eq_1" not in s:
            print(f"{name:34s} SKIPPED — {s.get('skipped')}")
            continue
        a, b = s["dt_eq_1"], s["dt_gt_1"]
        print(f"{name:34s} {a['temporal_error']['mean']:10.6f} "
              f"{a['do_nothing_anchor_unwarped']['mean']:9.6f} "
              f"{a['warp_gain_vs_unwarped']['mean']:6.2f} "
              f"{a['sharpness_ratio_vs_target']['mean']:7.4f} | "
              f"{b['temporal_error']['mean']:10.6f} "
              f"{b['sharpness_ratio_vs_target']['mean']:7.4f}")
    d = report["motion_response_diagnostic"]
    if d.get("sources"):
        print(f"\nmotion-response diagnostic — {d['n_pairs']} pairs >= 2 px, any split (NOT a gate):")
        for name, s2 in d["sources"].items():
            g = s2["dt_eq_1"] if s2["dt_eq_1"]["n_pairs"] else s2["dt_gt_1"]
            allr = [r for k in ("dt_eq_1", "dt_gt_1") for r in [s2[k]] if r["n_pairs"]]
            for r in allr:
                print(f"  {name:22s} n={r['n_pairs']:3d} TE={r['temporal_error']['mean']:.6f} "
                      f"anchor={r['do_nothing_anchor_unwarped']['mean']:.6f} "
                      f"gain={r['warp_gain_vs_unwarped']['mean']:.2f} "
                      f"median_motion={r['median_motion_px']['mean']:.2f} px")
            _ = g

    sp = report["cnn_across_seed_spread"].get("dt_eq_1_temporal_error")
    if sp:
        print(f"\nCNN across-seed TE spread (dt==1): mean {sp['mean']:.6f}  "
              f"sd {sp['sd']:.6f}  range {sp['range']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
