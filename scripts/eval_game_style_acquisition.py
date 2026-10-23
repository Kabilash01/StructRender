"""Style acquisition on the GAME models — the M4 criterion, never computed for them.

Same quantities and the same gate as `train_baseline.style_acquisition` (imported
unchanged from M4/M5, Decisions 35/41): colour-histogram travel away from the
native render, prediction vs teacher target (`stylisation_distance`), and
PSNR against the target versus the do-nothing native frame. Models run WITH
their one-frame history (eval_forward), on the sealed game test sessions, with
the Decision 94 frame exclusions — the frames Decision 95 scored.

Gate (unchanged): hist_pred_vs_native / hist_target_vs_native >= 0.80.
"""
from __future__ import annotations
import datetime as dt, json
import numpy as np, torch
from torch.utils.data import DataLoader
from data_pipeline.teacher.metrics import stylisation_distance
from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.image_quality import psnr
from renderer.temporal.pair_dataset import TemporalFrameDataset
from renderer.training import train_temporal as T
from renderer.training.dataset import target_dir_for
from renderer.training.train_baseline import ROOT, load_checkpoint

RUNS = [("two_style_B", "checkpoints/game_b_two_style/best.pt", "oil_impasto"),
        ("two_style_B", "checkpoints/game_b_two_style/best.pt", "watercolor_bold_s055"),
        ("oil_only_B", "checkpoints/game_b_oil/best.pt", "oil_impasto")]


def main() -> int:
    dev = torch.device("cuda")
    sp = json.load(open(ROOT / "configs/data/game_split_v1.json"))
    ex = json.load(open(ROOT / "configs/data/game_frame_exclusions_v1.json"))["excluded"]
    rep = {"gate": {"min_hist_travel_fraction_of_target": 0.80, "source": "M4/M5, unchanged (Decisions 35, 41)"},
           "test_sessions": sp["test_sessions"], "frame_exclusions": "configs/data/game_frame_exclusions_v1.json",
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "runs": []}
    for name, ck, style in RUNS:
        m = load_checkpoint(ROOT / ck, dev).eval()
        h_pn, h_tn, p_pt, p_nt = [], [], [], []
        for sid in sp["test_sessions"]:
            s = StructuralSession(ROOT / "data_pipeline/recorder" / sid)
            keep = [f for f in s.frame_ids() if str(f) not in ex.get(sid, {})]
            ds = TemporalFrameDataset(s, target_dir_for(sid, style), keep)
            for b in DataLoader(ds, batch_size=4, num_workers=6):
                fids = b["frame_id"].tolist()
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                if hasattr(m, "style_ids"):
                    b["style_idx"] = torch.full((len(fids),), m.style_ids.index(style), device=dev)
                with torch.no_grad():
                    pred = T.eval_forward(m, b).float().clamp(0, 1).cpu()
                y = b["target_curr"].float().cpu()
                p_pt.append(psnr(pred, y))
                for i, f in enumerate(fids):
                    native = s.load_native_rgb(f)
                    target = np.rint(y[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pu8 = np.rint(pred[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    nat = torch.from_numpy(native.astype(np.float32) / 255).permute(2, 0, 1)[None]
                    p_nt.append(psnr(nat, y[i:i + 1]))
                    h_pn.append(stylisation_distance(pu8, native)[1])
                    h_tn.append(stylisation_distance(target, native)[1])
        r = {"model": name, "style": style, "n_frames": len(h_pn),
             "hist_pred_vs_native": float(np.mean(h_pn)), "hist_target_vs_native": float(np.mean(h_tn)),
             "psnr_pred_vs_target_db": float(torch.cat(p_pt).mean()),
             "psnr_native_vs_target_db": float(torch.cat(p_nt).mean())}
        r["hist_travel_ratio"] = r["hist_pred_vs_native"] / r["hist_target_vs_native"]
        r["hist_pass"] = r["hist_travel_ratio"] >= 0.80
        r["beats_do_nothing_db"] = r["psnr_pred_vs_target_db"] - r["psnr_native_vs_target_db"]
        rep["runs"].append(r)
        print(json.dumps(r), flush=True)
    (ROOT / "eval/reports/game_style_acquisition.json").write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
