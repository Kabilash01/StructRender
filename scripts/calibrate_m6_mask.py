"""Calibrate the M6 confidence mask from measured error distributions — KB §4.

§4 gives ``M_t = exp(-alpha e_D - beta e_N)`` and leaves ``alpha`` and ``beta``
unspecified. Inventing them would make the mask an assertion; they are read off
**measured** distributions instead, and the measurement is committed so the
choice stays checkable.

Two things get measured, and the calibration needs both:

* **the error distributions** — what ``e_D`` (metres) and ``e_N`` (1 - cos)
  actually look like on the validation pairs, split by tick delta;
* **the scene's own geometry scale** — the depth step across a *within-surface*
  pixel edge versus across an *entity silhouette*, where the engine's entity-id
  map says the neighbour is a different object. That second pair of numbers is
  what makes the 0.5 crossing "a physically meaningful discontinuity" rather
  than a percentile of a distribution whose units nobody checked.

Writes ``eval/reports/m6_mask_calibration.json``. This script is its only writer.

    python -m scripts.calibrate_m6_mask
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from renderer.temporal.confidence import (
    ALPHA,
    BETA,
    E_D_HALF_METRES,
    E_N_HALF,
    PREV_COVERAGE_EPS,
    confidence_mask,
)
from renderer.temporal.pairs import (
    consecutive_pairs,
    entities_without_previous_transform,
    load_geometry,
    spawned_pixel_mask,
)
from renderer.temporal.warp import backward_warp, in_bounds
from renderer.training.dataset import discover_session, make_split
from renderer.training.train_baseline import ROOT, load_config

PERCENTILES = (1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 99.99)

#: Candidate half-mask points swept so the committed choice can be read against
#: its neighbours instead of appearing as the only value ever considered.
E_D_CANDIDATES = (0.02, 0.05, 0.10, 0.15, 0.175, 0.20, 0.25, 0.35, 0.50, 1.00, 2.00)
E_N_CANDIDATES_DEG = (10, 15, 20, 25, 30, 45, 60)

REQUIRED_TOP_LEVEL = (
    "milestone", "purpose", "formula", "generated_utc", "torch_version",
    "session_id", "style_id_note", "split", "motion_convention",
    "hard_zero_terms", "scene_geometry_scale", "error_distributions",
    "calibration", "sensitivity", "deviations_flagged_for_the_decision_log",
)


def _pcts(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "max": float(a.max()),
            **{f"p{q:g}": float(np.percentile(a, q)) for q in PERCENTILES}}


def _separation(within: np.ndarray, across: np.ndarray, taus) -> dict:
    """How well each candidate threshold separates within-surface from across-object.

    Youden's J = P(across > tau) - P(within > tau). This is the whole content of
    "physically meaningful": the crossing sits where a depth (or normal) step
    stops being explicable as the same surface and starts meaning a different
    object, judged on the engine's own labelling of which is which.
    """
    rows = {}
    for t in taus:
        tpr, fpr = float((across > t).mean()), float((within > t).mean())
        rows[f"{t:.5g}"] = {"across_object_above": tpr, "within_surface_above": fpr,
                            "youden_j": tpr - fpr}
    best = max(rows.items(), key=lambda kv: kv[1]["youden_j"])
    return {"sweep": rows, "argmax_tau": float(best[0]), "argmax_youden_j": best[1]["youden_j"]}


def scene_geometry_scale(session, frame_ids) -> dict:
    """The depth/normal step across a pixel edge, within a surface and across a silhouette.

    The silhouette set is selected by the engine's ``entity_id`` map — an
    **address**, used to find object boundaries, never fed anywhere as a value
    (KB Decision 9). It is the only non-circular way to ask "how big is a real
    occlusion boundary in this scene".
    """
    within, sil, nwithin, nsil, near, far = [], [], [], [], [], []
    for f in frame_ids:
        s = session.load_structure(f)
        d = np.asarray(s.depth, dtype=np.float32)
        n = np.asarray(s.normal, dtype=np.float32)
        e = np.asarray(s.entity_id)
        fin = np.isfinite(d)
        near.append(float(d[fin].min())), far.append(float(d[fin].max()))
        dz = np.nan_to_num(d, posinf=0.0)
        for step, ndot, ok_geom, same in (
            (np.abs(np.diff(dz, axis=1)), 1.0 - (n[:, 1:] * n[:, :-1]).sum(-1),
             fin[:, 1:] & fin[:, :-1], e[:, 1:] == e[:, :-1]),
            (np.abs(np.diff(dz, axis=0)), 1.0 - (n[1:, :] * n[:-1, :]).sum(-1),
             fin[1:, :] & fin[:-1, :], e[1:, :] == e[:-1, :]),
        ):
            within.append(step[ok_geom & same])
            sil.append(step[ok_geom & ~same])
            nwithin.append(ndot[ok_geom & same])
            nsil.append(ndot[ok_geom & ~same])
    within, sil = np.concatenate(within), np.concatenate(sil)
    nwithin, nsil = np.concatenate(nwithin), np.concatenate(nsil)
    return {
        "n_frames": len(frame_ids),
        "depth_range_m": {"near": min(near), "far": max(far)},
        "within_surface_neighbour_step_m": _pcts(within),
        "entity_silhouette_neighbour_step_m": _pcts(sil),
        "within_surface_neighbour_normal_error": _pcts(nwithin),
        "entity_silhouette_neighbour_normal_error": _pcts(nsil),
        "depth_separation": _separation(within, sil, E_D_CANDIDATES),
        "normal_separation": _separation(
            nwithin, nsil, [1.0 - float(np.cos(np.radians(x))) for x in E_N_CANDIDATES_DEG]),
        "note": ("A 'within-surface' step is a neighbouring pixel pair with the SAME "
                 "entity_id; a 'silhouette' step is one where the ids differ, i.e. the "
                 "engine says the neighbour is a different object. The crossing is placed "
                 "at the Youden-optimal separation of the two — see *_separation."),
    }


def measure(session, pairs) -> dict:
    """Error distributions and hard-zero statistics over a set of pairs."""
    ed, en, en_renorm, mask_vals = [], [], [], []
    terms = {"in_bounds": [], "bg_at_t": [], "prev_was_bg": [], "no_prev_transform": [],
             "hard_valid": []}
    s7_pairs = 0
    for p in pairs:
        c = load_geometry(session, p.curr_id)
        q = load_geometry(session, p.prev_id)
        m = c["motion"]
        fresh = entities_without_previous_transform(session, p.prev_id, p.curr_id)
        npt = spawned_pixel_mask(c["entity_id"], fresh) if fresh else None
        s7_pairs += 1 if fresh else 0

        out = confidence_mask(
            depth_t=c["depth_m"], depth_prev=q["depth_m"],
            normal_t=c["normal"], normal_prev=q["normal"], motion=m,
            depth_finite_t=c["depth_finite"], depth_finite_prev=q["depth_finite"],
            motion_valid=session.motion_valid(p.curr_id), no_prev_transform=npt)

        v = out["hard_valid"]
        sel = v > 0.5
        ed.append(out["e_d"][sel].numpy())
        en.append(out["e_n"][sel].numpy())
        wn = backward_warp(q["normal"], m)
        wn = wn / wn.norm(dim=1, keepdim=True).clamp_min(1e-6)
        en_renorm.append((1.0 - (c["normal"] * wn).sum(1, keepdim=True)).clamp(0, 2)[sel].numpy())
        mask_vals.append(out["mask"][sel].numpy())

        terms["in_bounds"].append(float(in_bounds(m).mean()))
        terms["bg_at_t"].append(float(1.0 - c["depth_finite"].mean()))
        terms["prev_was_bg"].append(float(
            (backward_warp(q["depth_finite"], m) <= 1.0 - PREV_COVERAGE_EPS).float().mean()))
        terms["no_prev_transform"].append(float(npt.mean()) if npt is not None else 0.0)
        terms["hard_valid"].append(float(v.mean()))

    ed = np.concatenate(ed) if ed else np.zeros(0, np.float32)
    en = np.concatenate(en) if en else np.zeros(0, np.float32)
    enr = np.concatenate(en_renorm) if en_renorm else np.zeros(0, np.float32)
    mv = np.concatenate(mask_vals) if mask_vals else np.zeros(0, np.float32)
    return {
        "n_pairs": len(pairs),
        "pairs_with_a_just_registered_entity": s7_pairs,
        "hard_zero_term_frame_fractions_mean": {k: float(np.mean(v)) if v else 0.0
                                                for k, v in terms.items()},
        "e_d_metres": _pcts(ed),
        "e_n_literal": _pcts(en),
        "e_n_renormalised_warped_normal": _pcts(enr),
        "mask_at_committed_alpha_beta": {
            "mean": float(mv.mean()) if mv.size else float("nan"),
            "frac_above_0.5": float((mv > 0.5).mean()) if mv.size else float("nan"),
            "frac_above_0.9": float((mv > 0.9).mean()) if mv.size else float("nan"),
            "note": "over hard-valid pixels only; hard-zeroed pixels are 0 by construction",
        },
        "_ed": ed, "_en": en,
    }


def sensitivity(groups: dict) -> dict:
    """What fraction of hard-valid pixels each candidate crossing would keep."""
    out: dict = {"e_d_half_metres": {}, "e_n_half_degrees": {}}
    for cand in E_D_CANDIDATES:
        a = float(np.log(2.0) / cand)
        out["e_d_half_metres"][f"{cand:g}"] = {
            g: float((np.exp(-a * m["_ed"]) > 0.5).mean()) if m["_ed"].size else None
            for g, m in groups.items()}
    for deg in E_N_CANDIDATES_DEG:
        half = 1.0 - float(np.cos(np.radians(deg)))
        b = float(np.log(2.0) / half)
        out["e_n_half_degrees"][str(deg)] = {
            "e_n_half": half,
            **{g: float((np.exp(-b * m["_en"]) > 0.5).mean()) if m["_en"].size else None
               for g, m in groups.items()}}
    out["note"] = ("Fraction of HARD-VALID pixels the soft term alone would keep above 0.5, "
                   "per candidate crossing. Committed values are "
                   f"e_D* = {E_D_HALF_METRES} m and e_N* = 1 - cos(30 deg).")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 confidence-mask calibration")
    ap.add_argument("--config", default="configs/training/m4_unet_baseline.json")
    ap.add_argument("--out", default="eval/reports/m6_mask_calibration.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    session = StructuralSession(cfg["data"].get("session_dir") or discover_session())
    split = make_split(session, cfg["split"]["val_sequences"], int(cfg["split"]["guard_frames"]))
    pairs = consecutive_pairs(session, split.val_ids)
    dt1 = [p for p in pairs if p.tick_delta == 1]
    dtn = [p for p in pairs if p.tick_delta > 1]

    print(f"[calib] {len(pairs)} val pairs — {len(dt1)} at dt==1, {len(dtn)} at dt>1", flush=True)
    groups = {}
    for name, ps in (("dt_eq_1", dt1), ("dt_gt_1", dtn)):
        print(f"[calib] measuring {name} ({len(ps)} pairs) ...", flush=True)
        groups[name] = measure(session, ps)

    report = {
        "milestone": 6,
        "purpose": ("Calibrate alpha/beta of KB §4's confidence mask from measured error "
                    "distributions and the scene's own geometry scale, rather than from "
                    "invented constants."),
        "formula": "M_t = hard_zeros * exp(-alpha*e_D - beta*e_N);  e_D = |D_t - W(D_{t-1})| "
                   "in metres;  e_N = 1 - N_t . W(N_{t-1})",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "session_id": session.session_id,
        "style_id_note": ("Style-independent by construction: every quantity here comes from "
                          "the engine's own depth/normal/motion buffers, never from a teacher "
                          "target or a model output."),
        "split": {
            "policy": split.as_report()["policy"],
            "val_sequences": split.val_sequences,
            "guard_frames": split.guard_frames,
            "n_val_frames": len(split.val_ids),
            "n_consecutive_val_pairs": len(pairs),
            "n_pairs_dt_eq_1": len(dt1),
            "n_pairs_dt_gt_1": len(dtn),
            "pairing_rule": ("(f-1, f) with BOTH ends inside the validation split. The two "
                             "sequence-start frames therefore contribute no pair — 166 val "
                             "frames give 164 pairs, not 166."),
            "tick_delta_note": ("Computed from each record's own 'tick' field, never assumed. "
                                "Consecutive frame_ids are always 1 apart; the SIMULATION time "
                                "between them is 1 or 8 ticks because the recorder drops to a "
                                "reduced rate during the long reappearance gaps."),
        },
        "motion_convention": ("Backward, screen-UV [0,1], Y-down, analytic from engine "
                             "transforms (KB Decision 25). Verified to compose: see "
                             "renderer/temporal/tests/test_warp.py."),
        "hard_zero_terms": {
            "list": ["motion_valid == False (frame 0)",
                     "sample out of bounds after warping",
                     "S7: entity with no t-1 transform (motion pinned to (0,0))",
                     "background/sky at t (depth == +inf, motion pinned to (0,0))",
                     "the warped source was background (added; see deviations)"],
            "why": ("Each is a case where the exponential is not merely large but meaningless: "
                    "the operands do not exist, or the motion value is bit-identical to a "
                    "correct one. S7 and background are the dangerous pair — both carry "
                    "motion (0,0), which the exponential reads as PERFECT confidence."),
            "prev_coverage_eps": PREV_COVERAGE_EPS,
        },
        "scene_geometry_scale": scene_geometry_scale(session, split.val_ids[::4]),
        "error_distributions": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                for k, v in groups.items()},
        "calibration": {
            "e_d_half_metres": E_D_HALF_METRES,
            "e_d_rationale": ("The Youden-optimal separation between the two measured "
                              "neighbour-step distributions in scene_geometry_scale: the depth "
                              "step at which 'same surface' stops explaining the difference and "
                              "'different object' starts. The separation curve is FLAT over "
                              "[0.10, 0.35] m, so the third digit carries no weight and the "
                              "value must not be re-tuned to move a result."),
            "e_n_half": E_N_HALF,
            "e_n_half_degrees": 30.0,
            "e_n_rationale": ("1 - cos(30 deg). A 30-degree orientation change is a change of "
                              "facet; correctly-tracked normals sit far below it (see "
                              "error_distributions), and a 90-degree cube edge lands at "
                              "2^-7.5 = 0.006, i.e. effectively zero. NOT LOAD-BEARING ON THIS "
                              "SCENE: the geometry is axis-aligned boxes and a ground plane, so "
                              "the normal separation is near-binary and its Youden index is flat "
                              "across 10-60 degrees (see scene_geometry_scale.normal_separation). "
                              "Curved geometry would make this term discriminative and would "
                              "need the calibration re-run."),
            "alpha": ALPHA, "beta": BETA,
            "derivation": "alpha = ln(2)/e_D*, beta = ln(2)/e_N* — each term alone crosses 0.5 "
                          "at its own half point, so both at their half points give 0.25.",
        },
        "sensitivity": sensitivity(groups),
        "deviations_flagged_for_the_decision_log": [
            {"item": "§4's e_D is undefined on background, on BOTH sides of the subtraction",
             "detail": ("depth carries +inf for background, so |D_t - W(D_{t-1})| is inf-inf = "
                        "NaN when either operand is sky. Zeroing only the current-frame side "
                        "leaves the 'was sky, is surface now' disocclusion undefined. A fifth "
                        "hard zero on the warped source's background coverage was added; it "
                        "is not in §4's three lines."),
             "action": "needs a Decision Log entry — NOT written by this run"},
            {"item": "e_N is not renormalised after warping",
             "detail": ("§4's form taken literally. Bilinear interpolation across a silhouette "
                        "shortens the warped normal and so raises e_N there, conflating "
                        "interpolation shortening with genuine orientation mismatch. Both "
                        "distributions are measured above; the difference is confined to the "
                        "top percentile."),
             "action": "needs a Decision Log entry — NOT written by this run"},
            {"item": "S7 never fires on this session",
             "detail": ("The hard zero for a just-registered entity is implemented and unit "
                        "tested, but this session registers both entities at frame 0 and spawns "
                        "nothing afterwards, so it has no data-backed exercise. Reported so it "
                        "is not mistaken for verified-in-the-field coverage."),
             "action": "needs a Decision Log entry — NOT written by this run"},
        ],
    }

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in report]
    if missing:
        raise SystemExit(f"generator/schema drift — REQUIRED_TOP_LEVEL not produced: {missing}")

    out = Path(ROOT) / args.out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[calib] wrote {out}")
    for g, m in report["error_distributions"].items():
        print(f"  {g}: pairs={m['n_pairs']} hard_valid={m['hard_zero_term_frame_fractions_mean']['hard_valid']:.3f} "
              f"e_D p95={m['e_d_metres']['p95']:.4f} p99={m['e_d_metres']['p99']:.4f} m | "
              f"e_N p95={m['e_n_literal']['p95']:.4f} p99={m['e_n_literal']['p99']:.4f} | "
              f"mask>0.5 on {m['mask_at_committed_alpha_beta']['frac_above_0.5']:.4f}")
    print(f"  alpha={ALPHA:.4f} (e_D*={E_D_HALF_METRES} m)  beta={BETA:.4f} (e_N*={E_N_HALF:.5f} = 30 deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
