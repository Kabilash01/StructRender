"""Pick the teacher style for the playable-game dataset — KB Decision 88.

The ladder's teacher runs at img2img strength 0.50, chosen (Decision 32) because
stronger settings invented geometry; the result is a style so light it is hard to
see (eval/showcase). The game is flat grey geometry, so there is more room for a
visible style — but "how far" is decided by the four alignment gates, not taste.

Grid: {oil_impasto, watercolor} x strength {0.50, 0.60, 0.70}, everything else as
committed. On a stride subset of one game TRAIN session:
  * gate failures (boundary F, spurious edges, silhouette support, stylisation)
  * stylisation strength = colour-histogram L1 travel from the native frame
Pick: the strongest setting with ZERO gate failures. It must then pass a FULL
session (Decision 40: cel_bold passed an 81-frame probe and failed 36/990).
"""
from __future__ import annotations
import dataclasses, datetime as dt, json, sys, time
import numpy as np
from data_pipeline.teacher.backends.sd15_controlnet import SD15ControlNetDepthBackend
from data_pipeline.teacher.metrics import Thresholds, frame_metrics, gate_failures, stylisation_distance
from data_pipeline.teacher.structural import StructuralSession
from data_pipeline.teacher.styles import load_style
from renderer.training.train_baseline import ROOT

STRENGTHS = tuple(float(x) for x in __import__("os").environ.get("SR_STRENGTHS", "0.50,0.60,0.70").split(","))
import os
STYLES = tuple(os.environ.get("SR_STYLES", "oil_impasto,watercolor").split(","))
STRIDE = 22   # 45 frames from a 990-frame session


def main() -> int:
    split = json.load(open(ROOT / "configs/data/game_split_v1.json"))
    s = StructuralSession(ROOT / "data_pipeline/recorder" / split["train_sessions"][0])
    frames = sorted(s.frame_ids())[::STRIDE]
    th = Thresholds()
    rep = {"purpose": "Pick the strongest teacher style that keeps all four alignment gates (Decision 88).",
           "session": s.session_id, "n_frames": len(frames), "stride": STRIDE,
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "grid": []}
    for style_id in STYLES:
        base = load_style(style_id)
        for st in STRENGTHS:
            params = dict(base.params); params["strength"] = st
            style = dataclasses.replace(base, params=params)
            be = SD15ControlNetDepthBackend(style)
            fails, hist, t0 = 0, [], time.time()
            for f in frames:
                nat = s.load_native_rgb(f); stc = s.load_structure(f)
                out = be.generate(nat, stc, style).styled_rgb
                m = frame_metrics(stc, out, th, native_rgb=nat); m["frame_id"] = f
                fails += bool(gate_failures(m, th))
                hist.append(stylisation_distance(out, nat)[1])
            row = {"style": style_id, "strength": st, "failing_frames": fails,
                   "stylisation_hist_mean": float(np.mean(hist)), "seconds": round(time.time() - t0)}
            rep["grid"].append(row)
            print(f"[style] {style_id:<12} strength {st:.2f}  failing {fails}/{len(frames)}  "
                  f"stylisation {row['stylisation_hist_mean']:.3f}", flush=True)
            del be
    ok = [r for r in rep["grid"] if r["failing_frames"] == 0]
    rep["pick"] = max(ok, key=lambda r: r["stylisation_hist_mean"]) if ok else None
    (ROOT / os.environ.get("SR_REPORT", "eval/reports/game_style_probe.json")).write_text(json.dumps(rep, indent=2) + "\n")
    print("[style] PICK:", rep["pick"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
