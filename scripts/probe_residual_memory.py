"""Probe: does the teacher carry information a MEMORY could hold? (pre-M8)

EPAM sharpens only if the part of the teacher that structure does NOT explain is
correlated across time for the same entity. If an entity looked a particular way
at t, memory lets the model predict that look at t+k instead of averaging over
everything it might look like — conditioning narrows the distribution and the
conditional mean gets sharper. That is the whole mechanism.

If the residual is INDEPENDENT frame to frame, memory carries zero information:
the distribution does not narrow, the L1+LPIPS optimum is the same blurry mean,
and EPAM cannot sharpen anything. The teacher generates each frame independently
by design (Decision 60), so this is a live possibility and must be measured
BEFORE M8, not discovered inside it.

Residual = teacher_t - model(structure_t), on the entity's own pixels. The model
supplies "what structure explains"; what is left is what a memory could carry.
No training.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import torch
from PIL import Image

from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.rcs import entity_mask_at
from renderer.training.dataset import build_input, target_dir_for
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, load_checkpoint, load_config

LAGS = (1, 2, 5, 10, 30)
MIN_PX = 200


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(ROOT / "configs/ladder_v3/model_a.json")
    model = load_checkpoint(ROOT / cfg["train"]["checkpoint_dir"] / "best.pt", dev).eval()
    split = load_session_split(ROOT / "configs/data/session_split_v2.json")

    same: dict[int, list] = {k: [] for k in LAGS}
    ctrl: list[float] = []          # different entities, same frame — the floor

    for sid in split["test_sessions"]:
        s = StructuralSession(ROOT / "data_pipeline" / "recorder" / sid)
        td = target_dir_for(s.session_id, "oil_impasto")
        idx = {int(json.loads(l)["frame_id"]): json.loads(l)["target_rgb_path"]
               for l in (td / "manifest.jsonl").read_text().splitlines() if l.strip()}
        ids = sorted(set(s.frame_ids()))

        def residual(f):
            x = torch.from_numpy(build_input(s.load_structure(f)))[None].to(dev)
            with torch.no_grad():
                p = model(x)[0].float().clamp(0, 1).cpu().numpy()
            pred = np.transpose(p, (1, 2, 0))
            tgt = np.asarray(Image.open(td / idx[f]).convert("RGB"), dtype=np.float32) / 255.0
            return tgt - pred

        cache: dict[int, np.ndarray] = {}
        def res(f):
            if f not in cache:
                if len(cache) > 60:
                    cache.clear()
                cache[f] = residual(f)
            return cache[f]

        # entities present often enough to follow
        ent = {}
        for f in ids[::10]:
            for iid in np.unique(s.load_structure(f).entity_id):
                if iid > 0:
                    ent.setdefault(int(iid), []).append(f)
        for iid, frames in ent.items():
            for lag in LAGS:
                for f in frames[:12]:
                    g = f + lag
                    if g not in set(ids):
                        continue
                    m1, m2 = entity_mask_at(s, f, iid), entity_mask_at(s, g, iid)
                    m = m1 & m2
                    if m.sum() < MIN_PX:
                        continue
                    a, b = res(f)[m], res(g)[m]
                    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
                    den = np.linalg.norm(a) * np.linalg.norm(b)
                    if den > 1e-8:
                        same[lag].append(float(np.dot(a, b) / den))
            # control: this entity vs a DIFFERENT entity in the same frame
            for f in frames[:8]:
                others = [j for j in ent if j != iid]
                if not others:
                    continue
                j = others[0]
                m1, m2 = entity_mask_at(s, f, iid), entity_mask_at(s, f, j)
                n = min(int(m1.sum()), int(m2.sum()))
                if n < MIN_PX:
                    continue
                a = res(f)[m1].ravel()[:n * 3]; b = res(f)[m2].ravel()[:n * 3]
                a, b = a - a.mean(), b - b.mean()
                den = np.linalg.norm(a) * np.linalg.norm(b)
                if den > 1e-8:
                    ctrl.append(float(np.dot(a, b) / den))
        print(f"[probe] {sid} done", flush=True)

    rep = {
        "question": ("Is the teacher residual (what structure does NOT explain) correlated "
                     "across time for the same entity? EPAM sharpens only if it is."),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "residual": "teacher_t - model_a_v3(structure_t), on the entity's own pixels",
        "same_entity_by_lag": {}, "control_different_entities": {},
    }
    print(f"\n{'lag':>5}{'corr':>10}{'sd':>9}{'n':>7}{'SE':>9}{'sigma':>8}")
    for lag in LAGS:
        v = np.asarray(same[lag])
        if v.size < 2:
            continue
        se = v.std(ddof=1) / np.sqrt(v.size)
        rep["same_entity_by_lag"][f"lag_{lag}"] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": int(v.size), "se": float(se)}
        print(f"{lag:>5}{v.mean():>10.4f}{v.std(ddof=1):>9.4f}{v.size:>7}{se:>9.4f}{abs(v.mean())/se:>8.1f}")
    c = np.asarray(ctrl)
    if c.size > 1:
        rep["control_different_entities"] = {
            "mean": float(c.mean()), "sd": float(c.std(ddof=1)), "n": int(c.size)}
        print(f"\n  control (different entities, same frame): {c.mean():+.4f}  n={c.size}")
    out = ROOT / "eval" / "reports" / "probe_residual_memory.json"
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"\n[probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
