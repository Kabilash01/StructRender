"""Re-tune the game styles at 768x432 to TODAY'S look — KB Decision 102 (option 1 of Decision 101).

SD 1.5 renders a different look at 768x432 with the committed settings (Decision
101). This probe picks, per style, the img2img strength whose 768x432 targets look
most like the committed 512x288 targets of the SAME frames — everything else (prompt,
seed, steps, guidance, ControlNet scale) as committed, one knob as in Decisions 88/92.

RULE, FIXED HERE BEFORE ANY RESULT:
  * data: a 768x432 recording of variant 1 (the route of ds_1790532989_38a03718),
    every 22nd frame (45 frames);
  * look match: mean LPIPS(alex) between the 768x432 target downscaled to 512x288
    (Lanczos) and the committed 512x288 target of the same frame — lower is closer;
  * eligible: zero gate failures on all 45 frames, gates at 768x432 with the
    Decision 100 pixel scaling;
  * pick: the eligible strength with the lowest mean LPIPS. No other criterion.
  * then: the pick must pass a FULL-session gate (Decision 40) before any bulk run.
Sanity: the two recordings must show the same frames — mean |native_768 downscaled
- native_512| is reported per probed frame.

Env: SR_HR_SESSION (the 768x432 session dir), SR_REPORT.
"""
from __future__ import annotations

import dataclasses, datetime as dt, json, os, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_pipeline.teacher.backends.sd15_controlnet import SD15ControlNetDepthBackend
from data_pipeline.teacher.metrics import Thresholds, frame_metrics, gate_failures, stylisation_distance
from data_pipeline.teacher.structural import StructuralSession
from data_pipeline.teacher.styles import load_style
from eval.metrics.image_quality import LpipsMetric
from renderer.training.dataset import target_dir_for
from renderer.training.train_baseline import ROOT

REF_SESSION = "ds_1790532989_38a03718"
GRID = {"oil_impasto": (0.50, 0.55, 0.60, 0.65, 0.70),
        "watercolor_bold_s055": (0.35, 0.40, 0.45, 0.50, 0.55)}
STRIDE = 22


def _down(a: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((512, 288), Image.LANCZOS), dtype=np.uint8)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None]


def main() -> int:
    hr = StructuralSession(Path(os.environ["SR_HR_SESSION"]))
    lo = StructuralSession(ROOT / "data_pipeline/recorder" / REF_SESSION)
    assert hr.resolution == [768, 432] or tuple(hr.resolution) == (768, 432), hr.resolution
    frames = sorted(hr.frame_ids())[::STRIDE]
    th = Thresholds().scaled_for((768, 432))
    # LPIPS on the CPU: it is tiny, and the GPU is shared with the desktop (a
    # first run hit a cuSOLVER handle failure with Chrome holding 2.5 GB).
    lp = LpipsMetric(device=torch.device("cpu"))
    sanity = {f: float(np.abs(_down(hr.load_native_rgb(f)).astype(float) - lo.load_native_rgb(f)).mean()) for f in frames}
    rep = {"purpose": "Strength per style whose 768x432 look best matches the committed 512x288 look (Decision 102).",
           "hr_session": str(hr.session_id), "ref_session": REF_SESSION, "frames": frames,
           "thresholds_768": th.as_dict(), "rule": "eligible = 0 gate failures on all frames; pick = lowest mean LPIPS",
           "native_correspondence_mean_abs_0_255": {"mean": float(np.mean(list(sanity.values()))),
                                                     "max": float(np.max(list(sanity.values())))},
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "grid": [], "pick": {}}
    print("[match] native correspondence mean|diff| (0-255):", rep["native_correspondence_mean_abs_0_255"], flush=True)
    for style_id, strengths in GRID.items():
        base = load_style(style_id)
        ref_dir = target_dir_for(REF_SESSION, style_id)
        rel = {json.loads(l)["frame_id"]: json.loads(l)["target_rgb_path"]
               for l in (ref_dir / "manifest.jsonl").read_text().splitlines() if l.strip()}
        for st in strengths:
            style = dataclasses.replace(base, params={**base.params, "strength": st})
            be = SD15ControlNetDepthBackend(style)
            fails, lps, hdist, styl, t0 = 0, [], [], [], time.time()
            for f in frames:
                nat, stc = hr.load_native_rgb(f), hr.load_structure(f)
                out = be.generate(nat, stc, style).styled_rgb
                m = frame_metrics(stc, out, th, native_rgb=nat)
                m["frame_id"] = f
                fails += bool(gate_failures(m, th))
                styl.append(stylisation_distance(out, nat)[1])
                ref = np.asarray(Image.open(ref_dir / rel[f]).convert("RGB"), dtype=np.uint8)
                d = _down(out)
                with torch.no_grad():
                    lps.append(float(lp(_t(d), _t(ref))))
                hdist.append(stylisation_distance(d, ref)[1])
            row = {"style": style_id, "strength": st, "failing_frames": fails,
                   "lpips_to_512_look": float(np.mean(lps)), "hist_to_512_look": float(np.mean(hdist)),
                   "stylisation_768": float(np.median(styl)), "seconds": round(time.time() - t0)}
            rep["grid"].append(row)
            print(f"[match] {style_id:<22} {st:.2f}  failing {fails}/{len(frames)}  LPIPS->512 {row['lpips_to_512_look']:.4f}"
                  f"  hist->512 {row['hist_to_512_look']:.3f}  stylisation {row['stylisation_768']:.3f}", flush=True)
            del be
            torch.cuda.empty_cache()
        ok = [r for r in rep["grid"] if r["style"] == style_id and r["failing_frames"] == 0]
        rep["pick"][style_id] = min(ok, key=lambda r: r["lpips_to_512_look"]) if ok else None
        print(f"[match] PICK {style_id}: {rep['pick'][style_id]}", flush=True)
    (ROOT / os.environ.get("SR_REPORT", "eval/reports/hr_style_match_probe.json")).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
