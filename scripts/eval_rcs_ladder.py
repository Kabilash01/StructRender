"""RCS on the trained ladder — the project's headline metric, measured at last.

RCS_i(dt) = cos(E(I_i^t), E(I_i^{t+dt})): does the SAME engine entity still look
like itself after leaving view and coming back? EPAM's whole claim (M8) is that
memory raises this, so the baselines' scores have to exist before the thing
meant to beat them is built.

Scored on the SEALED TEST sessions only. Every source — native frames, teacher
targets, each model — goes through one identical path, and the entity mask always
comes from the ENGINE's entity map, never from the render (KB Decision 9: IDs are
addresses). Each image metric is reported beside the do-nothing anchor, per §11.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.rcs import AppearanceEncoder, evaluate_source, reappearance_sequences, summarise
from renderer.training.dataset import build_input, target_dir_for
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, load_checkpoint, load_config

MODELS = {
    "a_v3": "configs/ladder_v3/model_a.json",
    "c_v3": "configs/ladder_v3/model_c.json",
    "a_v2": "configs/ladder_v2/model_a.json",
    "b_v2": "configs/ladder_v2/model_b.json",
    "c_v2": "configs/ladder_v2/model_c.json",
    "d_v2": "configs/ladder_v2/model_d.json",
}


def _model_loader(model, session, device):
    def load(fid):
        x = torch.from_numpy(build_input(session.load_structure(fid)))[None].to(device)
        with torch.no_grad():
            y = model(x)[0].float().clamp(0, 1).cpu().numpy()
        return np.transpose(y, (1, 2, 0))
    return load


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = AppearanceEncoder(device=device)
    split = load_session_split(ROOT / "configs/data/session_split_v2.json")
    report = {
        "metric": "RCS_i(dt) = cos(E(I_i^t), E(I_i^{t+dt})) over reappearance sequences",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "test_sessions": split["test_sessions"],
        "mask_source": "engine entity map (Decision 9: IDs are addresses, never inputs)",
        "sources": {},
    }
    rows: dict[str, list] = {}
    n_seq = 0

    for sid in split["test_sessions"]:
        session = StructuralSession(ROOT / "data_pipeline" / "recorder" / sid)
        seqs = reappearance_sequences(session)
        n_seq += len(seqs)
        if not seqs:
            continue
        tdir = target_dir_for(session.session_id, "oil_impasto")
        idx = {int(json.loads(l)["frame_id"]): json.loads(l)["target_rgb_path"]
               for l in (tdir / "manifest.jsonl").read_text().splitlines() if l.strip()}
        from PIL import Image

        def tload(fid, _t=tdir, _i=idx):
            return np.asarray(Image.open(_t / _i[fid]).convert("RGB"), dtype=np.float32) / 255.0

        def nload(fid, _s=session):
            return _s.load_native_rgb(fid).astype(np.float32) / 255.0

        rows.setdefault("native", []).extend(evaluate_source(session, nload, enc))
        rows.setdefault("teacher", []).extend(evaluate_source(session, tload, enc))

        for name, cfg_path in MODELS.items():
            cfg = load_config(ROOT / cfg_path)
            ck = ROOT / cfg["train"]["checkpoint_dir"] / "best.pt"
            if not ck.is_file():
                continue
            model = load_checkpoint(ck, device).eval()
            rows.setdefault(name, []).extend(
                evaluate_source(session, _model_loader(model, session, device), enc))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[rcs] {sid}: {len(seqs)} reappearance sequences", flush=True)

    report["n_reappearance_sequences"] = n_seq
    for k, v in rows.items():
        if v:
            report["sources"][k] = summarise(v)
    out = ROOT / "eval" / "reports" / "rcs_ladder.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[rcs] wrote {out}  ({n_seq} sequences)\n")
    anchor = report["sources"].get("native", {}).get("overall", {}).get("mean")
    print(f"{'source':<10}{'RCS':>10}{'sd':>9}{'n':>5}   vs native anchor")
    for k in ("native", "teacher", "a_v3", "c_v3", "a_v2", "b_v2", "c_v2", "d_v2"):
        if k not in report["sources"]:
            continue
        o = report["sources"][k]["overall"]
        sd = f"{o['sd']:.4f}" if o["sd"] is not None else "—"
        d = "" if k == "native" else f"{o['mean'] - anchor:+.4f}"
        print(f"{k:<10}{o['mean']:>10.4f}{sd:>9}{o['n']:>5}   {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
