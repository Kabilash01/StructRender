"""M6's actual metric on ladder v2 — KB Decision 66.

Temporal error: mask-weighted mean |I_t - W(I_{t-1}, V_t)| over the sealed TEST
sessions, restricted to pairs that actually move. This is the quantity §10's M6
exit criterion is about; everything else reported from ladder v2 is image quality.

Temporal rungs are rendered WITH their history, because scoring a temporal model
frame-independently measures a model that does not exist.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics as st

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.temporal import engine_confidence, evaluate_source, summarise
from renderer.temporal.pairs import consecutive_pairs
from renderer.training.dataset import build_input, target_dir_for
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, load_checkpoint, load_config

MODELS = {m: f"configs/ladder_v2/model_{m}.json" for m in ("a", "b", "c", "d")}
MOTION_PX = 1.0


def _native(session):
    return lambda fid: session.load_native_rgb(fid).astype(np.float32) / 255.0


def _plain(model, session, device):
    def load(fid):
        x = torch.from_numpy(build_input(session.load_structure(fid)))[None].to(device)
        with torch.no_grad():
            y = model(x)[0].float().clamp(0, 1).cpu().numpy()
        return np.transpose(y, (1, 2, 0))
    return load


def _temporal(model, session, device):
    """Render frame t from (S_t, encode(S_{t-1}), V_t, M_t) — as it runs."""
    ids = set(session.frame_ids())

    def load(fid):
        cur = torch.from_numpy(build_input(session.load_structure(fid)))[None].to(device)
        prev = fid - 1
        with torch.no_grad():
            if prev in ids and session.motion_valid(fid):
                pf = torch.from_numpy(build_input(session.load_structure(prev)))[None].to(device)
                feats, _ = model.encode(pf)
                from renderer.temporal.pairs import FramePair
                c = engine_confidence(session, FramePair(prev_id=prev, curr_id=fid, tick_delta=1))
                y = model(cur, feats, c["motion"].to(device), c["mask"].to(device))
            else:
                y = model(cur)
        return np.transpose(y[0].float().clamp(0, 1).cpu().numpy(), (1, 2, 0))
    return load


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_session_split()
    report = {
        "purpose": "M6's exit metric (temporal error) on ladder v2's sealed TEST sessions.",
        "metric": "mask-weighted mean |I_t - W(I_{t-1}, V_t)| over pairs with >= 1 px motion",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "test_sessions": split["test_sessions"], "motion_threshold_px": MOTION_PX,
        "temporal_rungs_rendered_with_history": True,
        "models": {},
    }
    acc: dict = {k: [] for k in list(MODELS) + ["native", "target"]}
    npairs = 0

    for sid in split["test_sessions"]:
        session = StructuralSession(ROOT / "data_pipeline" / "recorder" / sid)
        tdir = target_dir_for(session.session_id, "oil_impasto")
        pairs = [p for p in consecutive_pairs(session, session.frame_ids()) if p.tick_delta == 1]
        masks = {p.curr_id: engine_confidence(session, p) for p in pairs}
        idx = {int(json.loads(l)["frame_id"]): json.loads(l)["target_rgb_path"]
               for l in (tdir / "manifest.jsonl").read_text().splitlines() if l.strip()}
        from PIL import Image

        def tload(fid, _t=tdir, _i=idx):
            return np.asarray(Image.open(_t / _i[fid]).convert("RGB"), dtype=np.float32) / 255.0

        nat = summarise(evaluate_source(session, _native(session), pairs, tload, masks))
        moving = {q["curr_id"] for q in nat["per_pair"] if (q.get("median_motion_px") or 0) >= MOTION_PX}
        npairs += len(moving)
        acc["native"] += [q["te"] for q in nat["per_pair"] if q["curr_id"] in moving]
        tgt = summarise(evaluate_source(session, tload, pairs, tload, masks))
        acc["target"] += [q["te"] for q in tgt["per_pair"] if q["curr_id"] in moving]

        for name, cfg_path in MODELS.items():
            cfg = load_config(ROOT / cfg_path)
            ck = ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
            if not ck.is_file():
                continue
            model = load_checkpoint(ck, device).eval()
            loader = _temporal(model, session, device) if hasattr(model, "temporal_fuse") \
                else _plain(model, session, device)
            s = summarise(evaluate_source(session, loader, pairs, tload, masks))
            acc[name] += [q["te"] for q in s["per_pair"] if q["curr_id"] in moving]
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[te] {sid}: {len(moving)} moving pairs", flush=True)

    report["n_moving_pairs"] = npairs
    for k, v in acc.items():
        if v:
            report["models"][k] = {"n": len(v), "mean": st.mean(v),
                                   "sd": st.stdev(v) if len(v) > 1 else None}
    out = ROOT / "eval" / "reports" / "ladder_v2_temporal_error.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[te] wrote {out}\n")
    for k in ("native", "target", "a", "b", "c", "d"):
        if k in report["models"]:
            print(f"  {k:<8} TE {report['models'][k]['mean']:.6f}  (n={report['models'][k]['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
