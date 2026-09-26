"""Probe: can the teacher give each entity a CONSISTENT appearance over time?
(KB Decision 81; follows Decision 79.)

Why this is the question. EPAM remembers an entity's appearance and restores it.
That only helps if the training targets HAVE a persistent per-entity appearance.
The teacher currently draws one fixed noise field pinned to the SCREEN
(`sd15_controlnet.py`: one generator per style, not per frame), so an entity
that moves slides through different noise and is re-invented each frame. Memory
then has nothing to recover — consistent with Decisions 72, 73 and 79.

Condition A (baseline): the teacher exactly as it runs today.
Condition B (anchored): identical, except each entity's latent cells take noise
from a field owned by that entity — seeded from its EntityKey, mapped through its
bounding box — so the entity carries the same noise wherever it appears, and gets
it back after it leaves view. Background keeps the screen-pinned field.
This differs from Decision 60's refuted approach, which chained OUTPUTS frame to
frame; here no generated pixel feeds another frame.

Measured on the same frames for A, B and the native engine render:
  * RCS on reappearance sequences (does the entity look the same after the gap?)
  * temporal error on a contiguous run (flicker)
  * the four alignment gates on every generated frame (anchoring must not
    buy consistency by breaking geometry)

Runs on CPU (the GPU is training). Deliberately small: a probe decides whether a
full EPAM v2 is worth building, it is not the result itself.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from data_pipeline.teacher.backends import sd15_controlnet as SDB
from data_pipeline.teacher.metrics import Thresholds, frame_metrics, gate_failures
from data_pipeline.teacher.structural import StructuralSession
from data_pipeline.teacher.styles import load_style
from eval.metrics.rcs import AppearanceEncoder, _visible_run, entity_mask_at, rcs, reappearance_sequences
from renderer.entity.keys import keys_for_frame
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT

LAT = 8          # SD VAE downsampling
FIELD = 16       # per-entity noise field resolution
WINDOW = 2       # frames each side of a reappearance gap
RUN = 6          # contiguous frames for the flicker measurement

_NOISE = {"override": None}


def _patched_randn(shape, generator=None, device=None, dtype=None, layout=None):
    n = _NOISE["override"]
    if n is not None and tuple(n.shape) == tuple(shape):
        return n.to(device=device, dtype=dtype)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def anchored_noise(style_seed, entity_id_map, entities, shape):
    """Screen-pinned base noise (what the teacher uses today), with each entity's
    latent cells replaced from that entity's own field."""
    g = torch.Generator().manual_seed(int(style_seed))
    base = torch.randn(shape, generator=g)
    keys = keys_for_frame(entities)
    _, c, h, w = shape
    ids = torch.from_numpy(entity_id_map.astype(np.int64))
    blocks = ids[: h * LAT, : w * LAT].reshape(h, LAT, w, LAT).permute(0, 2, 1, 3).reshape(h, w, -1)
    owner = torch.mode(blocks, dim=-1).values                       # majority entity per latent cell
    for iid, key in keys.items():
        cells = (owner == iid).nonzero()
        if len(cells) == 0:
            continue
        y0, x0 = cells.min(0).values.tolist(); y1, x1 = cells.max(0).values.tolist()
        fg = torch.Generator().manual_seed(zlib.crc32(str(key).encode()) & 0x7FFFFFFF)
        field = torch.randn((c, FIELD, FIELD), generator=fg)
        for y, x in cells.tolist():
            u = min(FIELD - 1, int((y - y0 + 0.5) / (y1 - y0 + 1) * FIELD))
            v = min(FIELD - 1, int((x - x0 + 0.5) / (x1 - x0 + 1) * FIELD))
            base[0, :, y, x] = field[:, u, v]
    return base


def main() -> int:
    t0 = time.time()
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 4))
    sp = load_session_split(ROOT / "configs/data/session_split_v2.json")
    sid = sp["test_sessions"][0]
    s = StructuralSession(ROOT / "data_pipeline" / "recorder" / sid)
    style = load_style("oil_impasto")
    ents = {}
    for line in open(os.path.join(str(s.dir), "manifest.jsonl")):
        if line.strip():
            r = json.loads(line); ents[int(r["frame_id"])] = r.get("entities", [])

    # frames the measurements need
    seqs = []
    for q in reappearance_sequences(s):
        iid = int(q["target_instance_id"])
        b = _visible_run(s, iid, int(q["last_seen_frame"]), -1, WINDOW, 200)
        a = _visible_run(s, iid, int(q["first_seen_again_frame"]), +1, WINDOW, 200)
        if b and a:
            seqs.append({"id": q["sequence_id"], "iid": iid, "before": b, "after": a,
                         "unseen_seconds": float(q["unseen_seconds"])})
    temporal = [t for t in s.header["sequences"] if t["sequence_kind"] == "temporal"]
    t_start = int(temporal[0]["frame_range"][0]) if temporal else sorted(s.frame_ids())[100]
    run = [t_start + 10 + i for i in range(RUN)]
    frames = sorted({f for q in seqs for f in q["before"] + q["after"]} | set(run))
    print(f"[probe] session {sid}: {len(seqs)} reappearance sequences, {len(frames)} frames x 2 conditions", flush=True)

    SDB.randn_tensor = _patched_randn            # module-level name used by our backend? (pipeline below)
    import diffusers.pipelines.controlnet.pipeline_controlnet_img2img as P
    P.randn_tensor = _patched_randn
    be = SDB.SD15ControlNetDepthBackend(style)
    th = Thresholds()
    imgs = {"A": {}, "B": {}}
    gates = {"A": [], "B": []}
    shape = (1, 4, 288 // LAT, 512 // LAT)
    for n, f in enumerate(frames):
        nat = s.load_native_rgb(f); st = s.load_structure(f)
        for cond in ("A", "B"):
            _NOISE["override"] = None if cond == "A" else anchored_noise(style.seed, st.entity_id, ents.get(f, []), shape)
            out = be.generate(nat, st, style).styled_rgb
            imgs[cond][f] = out
            m = frame_metrics(st, out, th, native_rgb=nat); m["frame_id"] = f
            gates[cond].append(len(gate_failures(m, th)))
        if n % 5 == 0:
            print(f"[probe] {n+1}/{len(frames)} frames  {time.time()-t0:.0f}s", flush=True)
    _NOISE["override"] = None

    enc = AppearanceEncoder(device=torch.device("cpu"))
    def as_f(img): return img.astype(np.float32) / 255.0
    sources = {"native": lambda f: as_f(s.load_native_rgb(f)),
               "A_screen_noise": lambda f: as_f(imgs["A"][f]),
               "B_anchored_noise": lambda f: as_f(imgs["B"][f])}
    out = {"purpose": "Can the teacher give each entity a consistent appearance over time? (Decision 81)",
           "session": sid, "style": "oil_impasto", "n_frames": len(frames),
           "n_sequences": len(seqs), "device": "cpu",
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rcs": {}, "temporal_error": {}, "gate_failing_frames": {}}
    for name, fn in sources.items():
        scores = []
        for q in seqs:
            eb = [enc(fn(f), entity_mask_at(s, f, q["iid"])) for f in q["before"]]
            ea = [enc(fn(f), entity_mask_at(s, f, q["iid"])) for f in q["after"]]
            scores.append(float(np.mean([rcs(x, y) for x in eb for y in ea])))
        out["rcs"][name] = {"mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)),
                            "per_sequence": scores}
        diffs = [float(np.abs(fn(run[i + 1]) - fn(run[i])).mean()) for i in range(len(run) - 1)]
        out["temporal_error"][name] = {"mean_abs_frame_diff": float(np.mean(diffs))}
    for cond in ("A", "B"):
        out["gate_failing_frames"][cond] = int(sum(1 for g in gates[cond] if g))
    a, b = out["rcs"]["A_screen_noise"]["per_sequence"], out["rcs"]["B_anchored_noise"]["per_sequence"]
    d = np.array(b) - np.array(a)
    out["rcs_B_minus_A"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                            "sigma": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))}
    (ROOT / "eval/reports/probe_anchored_noise.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n[probe] RCS on reappearance (higher = entity looks the same after the gap)")
    for k, v in out["rcs"].items(): print(f"   {k:<18} {v['mean']:.4f}  sd {v['sd']:.4f}")
    r = out["rcs_B_minus_A"]; print(f"   B - A paired: {r['mean']:+.4f}  SE {r['se']:.4f}  -> {r['sigma']:.1f} sigma")
    print("[probe] flicker on a contiguous run (mean |frame_t - frame_t-1|, lower = steadier)")
    for k, v in out["temporal_error"].items(): print(f"   {k:<18} {v['mean_abs_frame_diff']:.4f}")
    print(f"[probe] frames failing any alignment gate: A {out['gate_failing_frames']['A']}  B {out['gate_failing_frames']['B']}  (of {len(frames)})")
    print(f"[probe] wall {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
