"""M9 temporal error on ladder v3 — the metric the temporal rungs exist for.

Temporal error: mask-weighted mean |I_t - W(I_{t-1}, V_t)| over the sealed TEST
sessions, on pairs that actually move (>= 1 px median motion). Every rung is
rendered with the inputs it runs with: B and D with their real previous frame,
E with its engine masks, A and C frame by frame.

Noise yardstick, fixed here BEFORE any number exists: Model E's three seeds give
the run-to-run spread of temporal error on this data. A temporal rung "improves"
on its no-history partner only if its temporal error is lower by more than two
standard deviations of a one-run-vs-one-run difference: 2*sqrt(2)*sd_rel. That
applies to D vs C (M6's question, replicated on v3) and to B vs A — the first
test of CNN temporal conditioning ever run, because B was never trained with
history before Decision 80.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics as st

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.temporal import engine_confidence, evaluate_source, summarise
from renderer.entity.keys import keys_for_frame
from renderer.entity.selection import build_entity_masks, pixel_counts, select_entities
from renderer.temporal.pairs import FramePair, consecutive_pairs
from renderer.training.dataset import build_input, target_dir_for
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, load_checkpoint

MODELS = {"a": "ladder_v3_a", "b": "ladder_v3_b", "c": "ladder_v3_c", "d": "ladder_v3_d",
          "e": "ladder_v3_e", "e_seedB": "ladder_v3_e_seedB", "e_seedC": "ladder_v3_e_seedC"}
MOTION_PX = 1.0


def _render(model, session, device, kind, ents):
    ids = set(session.frame_ids())

    def load(fid):
        st_ = session.load_structure(fid)
        x = torch.from_numpy(build_input(st_))[None].to(device)
        with torch.no_grad():
            if kind == "temporal":
                prev = fid - 1
                if prev in ids and session.motion_valid(fid):
                    pf = torch.from_numpy(build_input(session.load_structure(prev)))[None].to(device)
                    feats, _ = model.encode(pf)
                    c = engine_confidence(session, FramePair(prev_id=prev, curr_id=fid, tick_delta=1))
                    y = model(x, feats, c["motion"].to(device), c["mask"].to(device))
                else:
                    y = model(x)
            elif kind == "entity":
                keys = keys_for_frame(ents.get(fid, []))
                sel = select_entities(pixel_counts(st_.entity_id), keys, 8)
                m = torch.from_numpy(build_entity_masks(st_.entity_id, sel, 8))[None].to(device)
                y = model(x, m)
            else:
                y = model(x)
        return np.transpose(y[0].float().clamp(0, 1).cpu().numpy(), (1, 2, 0))
    return load


def _kind(model) -> str:
    if hasattr(model, "temporal_fuse"):
        return "temporal"
    return "entity" if type(model).__name__ == "EntityAwareSingleStyle" else "plain"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_session_split(ROOT / "configs/data/session_split_v2.json")
    acc = {k: [] for k in list(MODELS) + ["native", "target"]}
    kinds = {}
    for sid in split["test_sessions"]:
        s = StructuralSession(ROOT / "data_pipeline" / "recorder" / sid)
        tdir = target_dir_for(s.session_id, "oil_impasto")
        ents = {}
        for line in open(str(s.dir) + "/manifest.jsonl"):
            if line.strip():
                r = json.loads(line); ents[int(r["frame_id"])] = r.get("entities", [])
        pairs = [p for p in consecutive_pairs(s, s.frame_ids()) if p.tick_delta == 1]
        masks = {p.curr_id: engine_confidence(s, p) for p in pairs}
        idx = {int(json.loads(l)["frame_id"]): json.loads(l)["target_rgb_path"]
               for l in (tdir / "manifest.jsonl").read_text().splitlines() if l.strip()}
        from PIL import Image

        def tload(fid, _t=tdir, _i=idx):
            return np.asarray(Image.open(_t / _i[fid]).convert("RGB"), dtype=np.float32) / 255.0

        nat = summarise(evaluate_source(s, lambda f: s.load_native_rgb(f).astype(np.float32) / 255.0,
                                        pairs, tload, masks))
        moving = {q["curr_id"] for q in nat["per_pair"] if (q.get("median_motion_px") or 0) >= MOTION_PX}
        acc["native"] += [q["te"] for q in nat["per_pair"] if q["curr_id"] in moving]
        tg = summarise(evaluate_source(s, tload, pairs, tload, masks))
        acc["target"] += [q["te"] for q in tg["per_pair"] if q["curr_id"] in moving]
        for name, ck in MODELS.items():
            model = load_checkpoint(ROOT / "checkpoints" / ck / "best.pt", device).eval()
            kinds[name] = _kind(model)
            r = summarise(evaluate_source(s, _render(model, s, device, kinds[name], ents), pairs, tload, masks))
            acc[name] += [q["te"] for q in r["per_pair"] if q["curr_id"] in moving]
            del model; torch.cuda.empty_cache()
        print(f"[te] {sid}: {len(moving)} moving pairs", flush=True)

    rep = {"purpose": "M9 temporal error, ladder v3, sealed test split, every rung with its real inputs.",
           "metric": "mask-weighted mean |I_t - W(I_{t-1}, V_t)|, pairs with >= 1 px median motion",
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "test_sessions": split["test_sessions"], "inputs": kinds, "models": {}}
    for k, v in acc.items():
        rep["models"][k] = {"n": len(v), "mean": st.mean(v)}
    es = [rep["models"][k]["mean"] for k in ("e", "e_seedB", "e_seedC")]
    sd_rel = st.stdev(es) / st.mean(es)
    thr = 2 * math.sqrt(2) * sd_rel
    rep["noise"] = {"e_seeds_te": es, "sd_rel": sd_rel, "threshold_rel_one_run_each": thr,
                    "rule": "a temporal rung improves only if TE is lower than its partner's by > threshold"}
    for t, base in (("d", "c"), ("b", "a")):
        d = (rep["models"][t]["mean"] - rep["models"][base]["mean"]) / rep["models"][base]["mean"]
        rep[f"{t}_vs_{base}"] = {"rel_change": d, "threshold": thr,
                                 "verdict": "IMPROVES" if d < -thr else ("WORSE" if d > thr else "within noise")}
    (ROOT / "eval/reports/m9_temporal_error.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("\n[te] temporal error (lower = steadier):")
    for k in ("native", "target", "a", "b", "c", "d", "e", "e_seedB", "e_seedC"):
        print(f"   {k:<8} {rep['models'][k]['mean']:.6f}  (n={rep['models'][k]['n']})")
    print(f"[te] noise: E seed sd {100*sd_rel:.2f}% -> threshold {100*thr:.2f}%")
    for t, base in (("d", "c"), ("b", "a")):
        r = rep[f"{t}_vs_{base}"]; print(f"[te] {t.upper()} vs {base.upper()}: {100*r['rel_change']:+.2f}%  -> {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
