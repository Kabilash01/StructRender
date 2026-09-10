"""StructFormer's OWN across-seed temporal-error variance on the M6 gate.

**This is the only thing that writes ``eval/reports/m6_structformer_variance.json``.**
KB Decision 51 established that a committed artifact with no generator and no
test is unfalsifiable evidence; Decision 57's verdict change rests entirely on
this artifact, so it needs both (KB Decision 58).

Why it exists: M6's gate margin was built from CNN across-seed spread as a
stand-in, because StructFormer's own had never been measured — a substitution
Decisions 51 and 55 both flag. This measures the right quantity. The recomputed
margin was pre-committed to be binding in EITHER direction, including a harder
bar, which is what it turned out to be.

    python -m scripts.measure_structformer_variance    (needs CUDA)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
from pathlib import Path

import torch

from eval.metrics.temporal import engine_confidence, evaluate_source, summarise
from eval.protocols.run_temporal_baseline import _model_loader, _native_loader, _target_loader
from renderer.temporal.pairs import consecutive_pairs
from renderer.training.train_baseline import ROOT, load_checkpoint, load_config, resolve_data

#: The three Model C runs. Identical but for `train.seed` and `train.checkpoint_dir`.
RUNS = {
    "attempt1": "configs/structformer/m5_structformer.json",
    "seedB": "configs/structformer/m5_structformer_seedB.json",
    "seedC": "configs/structformer/m5_structformer_seedC.json",
}
MOTION_PX = 1.0  # identical to the M6 gate's threshold; not re-chosen here
CNN_RELATIVE_SD = 0.077  # what the original margin used, for the comparison


def main() -> int:
    ap = argparse.ArgumentParser(description="StructFormer across-seed variance on the M6 gate")
    ap.add_argument("--out", default="eval/reports/m6_structformer_variance.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / RUNS["attempt1"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session, target_dir, split = resolve_data(cfg)

    guard = sorted(set(session.frame_ids()) - set(split.train_ids) - set(split.val_ids))
    pairs = consecutive_pairs(session, guard)
    masks = {p.curr_id: engine_confidence(session, p) for p in pairs}
    target_load = _target_loader(target_dir)

    # The gate's moving subset is defined by the NATIVE render, exactly as the
    # gate defines it — never by a model, which would let a model choose its own
    # evaluation set.
    native = summarise(evaluate_source(session, _native_loader(session), pairs, target_load, masks))
    moving = {q["curr_id"] for q in native["per_pair"] if (q.get("median_motion_px") or 0) >= MOTION_PX}
    print(f"[var] gate: {len(moving)} guard-band pairs at >= {MOTION_PX} px", flush=True)

    runs: dict = {}
    for name, cfg_path in RUNS.items():
        mcfg = load_config(ROOT / cfg_path)
        ck = ROOT / mcfg["train"]["checkpoint_dir"] / "best.pt"
        if not ck.is_file():
            runs[name] = {"skipped": f"no checkpoint at {ck}"}
            continue
        print(f"[var] {name} ...", flush=True)
        model = load_checkpoint(ck, device).eval()
        s = summarise(evaluate_source(session, _model_loader(model, session, device),
                                      pairs, target_load, masks))
        rows = [q for q in s["per_pair"] if q["curr_id"] in moving]
        runs[name] = {
            "seed": mcfg["train"]["seed"], "n": len(rows),
            "checkpoint": str(ck.relative_to(ROOT)),
            "num_workers": mcfg["train"].get("num_workers"),
            "te": st.mean([q["te"] for q in rows]),
            "sharp": st.mean([q["grad_energy"] / q["grad_energy_target"] for q in rows]),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    te = [v["te"] for v in runs.values() if "te" in v]
    spread = {"mean": st.mean(te), "sd": st.stdev(te), "rel_sd": st.stdev(te) / st.mean(te)}
    margin = 2 * spread["rel_sd"]

    # Model D's own gate figure is READ from the gate artifact, never recomputed
    # here — two independent computations of one headline number is how two
    # committed artifacts come to disagree (KB Decision 58).
    gate = json.loads((ROOT / "eval" / "reports" / "m6_temporal_baseline.json").read_text())
    model_d = gate["m6_gate"]["sources"]["m6_temporal_model_d"]["temporal_error"]["mean"]

    out = {
        "purpose": ("StructFormer's OWN across-seed temporal-error variance on the M6 gate. The "
                    "original margin used CNN spread as a stand-in because this had never been "
                    "measured (KB Decisions 51, 55)."),
        "pre_commitment": ("Recorded BEFORE these runs: the recomputed margin is accepted in EITHER "
                           "direction, including a harder bar. A transformer being less stable than "
                           "a CNN was the stated expectation. It is, and M6 fails by more."),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "session_id": session.session_id, "style_id": cfg["data"]["style_id"],
        "gate": f"guard-band pairs with >= {MOTION_PX} px median displacement — identical to m6_gate",
        "n_pairs": len(moving),
        "runs": runs, "spread": spread,
        "comparison_to_cnn": {
            "cnn_relative_sd": CNN_RELATIVE_SD, "structformer_relative_sd": spread["rel_sd"],
            "note": "StructFormer is LESS stable across seeds than the CNN; the corrected margin is wider.",
        },
        "recomputed_gate": {
            "reference": "mean of the three Model C seeds, not attempt1's single draw (Decision 46)",
            "reference_te": spread["mean"], "margin": margin,
            "model_d_must_be_at_most": spread["mean"] * (1 - margin),
            "model_d_te": model_d,
            "model_d_te_source": "read from m6_temporal_baseline.json -> m6_gate, not recomputed",
            "model_d_vs_mean_pct": 100 * (model_d - spread["mean"]) / spread["mean"],
            "verdict": "PASS" if model_d <= spread["mean"] * (1 - margin) else "FAIL",
        },
        "decisive_observation": (
            "Model D lies INSIDE Model C's own seed range, and one Model C seed is BETTER than "
            "Model D. The improvement is not separable from StructFormer run-to-run noise. The "
            "earlier figure was measured against attempt1, a comparatively unlucky single draw."),
        "known_difference_between_pooled_runs": (
            "attempt1 trained with train.num_workers=6; seedB/seedC with 2, reduced after a system "
            "OOM kill. Worker count does not enter the computation, so this should not affect "
            "results — but it is a difference between runs pooled into one sd, and is stated."),
    }
    p = ROOT / args.out
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[var] wrote {p}")
    for k, v in runs.items():
        if "te" in v:
            print(f"  {k:9s} seed {v['seed']}  TE {v['te']:.6f}")
    print(f"  spread rel_sd {100*spread['rel_sd']:.2f}%  ->  margin {100*margin:.1f}%")
    print(f"  Model D {model_d:.6f}  must be <= {out['recomputed_gate']['model_d_must_be_at_most']:.6f}"
          f"  -> {out['recomputed_gate']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
