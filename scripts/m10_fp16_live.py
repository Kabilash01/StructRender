"""M10 deliverable "TensorRT FP16 engine" for the live temporal step (Decision 98).

FP16 on TensorRT 11 means an FP16 ONNX graph in a STRONGLY_TYPED network
(Decision 45 item 1). The confidence mask stays FP32 (depth in metres needs it;
1/x0 - 1 at fp16 would err by ~5 cm at 50 m against a 0.175 m mask scale); the
network runs FP16; the engine's I/O stays float32, so the C++ runtime is
unchanged.

FP16 TOLERANCE — FIXED HERE BEFORE ANY FP16 NUMBER WAS MEASURED (Decision 45
deferred it to M10). The FP32 1e-3 max-abs gate cannot apply: half precision
carries ~1e-3 relative error by construction. The question that matters is
whether FP16 changes what is scored, so on 100 real test pairs per style
(every 30th frame of the 3 sealed test sessions, real one-frame history),
FP16 engine vs strict-FP32 engine:
  (a) mean |PSNR_fp16 - PSNR_fp32| against the teacher target < 0.05 dB, and
  (b) mean absolute pixel difference between the two outputs < 1/255.
FP16 PASSES only if (a) and (b) hold in BOTH styles. It SHIPS only if it also
passes and is faster than strict FP32 — otherwise strict FP32 stays (Decision 97).
"""
from __future__ import annotations

import copy, datetime as dt, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.image_quality import psnr
from inference.export.trt_compat import for_tensorrt
from inference.tensorrt.builder.build_engine import build_engine
from renderer.temporal.live_step import LiveStep
from renderer.temporal.pair_dataset import TemporalFrameDataset
from renderer.training.dataset import target_dir_for
from renderer.training.train_baseline import ROOT, load_checkpoint
from scripts.m10_cpp_parity import ENGINES, SRT
from scripts.m10_live_temporal_engines import NAMES_IN, NAMES_OUT, real_inputs

TOL_DPSNR, TOL_MAD = 0.05, 1.0 / 255


class LiveStepFP16(LiveStep):
    """Mask in fp32, network in fp16, float32 in and out."""

    def forward(self, x_t, x_prev, feat_prev, motion, no_prev, motion_valid):
        mask = self.confidence(x_t, x_prev, motion, no_prev, motion_valid)
        feat, skips = self.model.encode(x_t.half())
        fused = self.model.fuse(feat, feat_prev.half(), motion.half(), mask.half())
        style = torch.full((x_t.shape[0],), self.style_idx, dtype=torch.long, device=x_t.device)
        return self.model.decode(fused, skips, style).float(), feat.float()


class TRT:
    def __init__(self, path):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.eng = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]

    def __call__(self, ins: dict):
        outs = {n: torch.empty(tuple(self.eng.get_tensor_shape(n)), device="cuda") for n in NAMES_OUT}
        bufs = {n: ins[n].float().contiguous().cuda() for n in NAMES_IN}
        for n in self.names:
            self.ctx.set_tensor_address(n, (bufs.get(n) if n in bufs else outs[n]).data_ptr())
        s = torch.cuda.Stream()
        self.ctx.execute_async_v3(s.cuda_stream); s.synchronize()
        return outs["rgb"], outs["feat"]


def bench(eng):
    _, ins = real_inputs(load_checkpoint(ROOT / "checkpoints/game_b_two_style/best.pt", torch.device("cpu")).eval())
    tmp = Path(tempfile.mkdtemp()); files = []
    for n, t in zip(NAMES_IN, ins):
        (tmp / f"{n}.bin").write_bytes(t.float().numpy().tobytes()); files.append(str(tmp / f"{n}.bin"))
    r = subprocess.run([str(SRT), str(eng), *files, "--out", str(tmp / "o.bin"), "--bench", "500"], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.startswith("bench")][0], tmp


def main() -> int:
    dev = torch.device("cuda")
    m = load_checkpoint(ROOT / "checkpoints/game_b_two_style/best.pt", dev).eval()
    mh = copy.deepcopy(m).half()
    _, ins_cpu = real_inputs(load_checkpoint(ROOT / "checkpoints/game_b_two_style/best.pt", torch.device("cpu")).eval())
    ins = [t.to(dev) for t in ins_cpu]
    rep = {"tolerance_preregistered": {"mean_abs_dpsnr_db": TOL_DPSNR, "mean_abs_pixel_diff": TOL_MAD,
                                       "pairs_per_style": "every 30th frame of the 3 test sessions"},
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "styles": {}}
    sp = json.load(open(ROOT / "configs/data/game_split_v1.json"))
    ex = json.load(open(ROOT / "configs/data/game_frame_exclusions_v1.json"))["excluded"]
    passed = True
    for idx, style in enumerate(m.style_ids):
        onnx_p = Path(tempfile.mkdtemp()) / f"live16_{style}.onnx"
        with torch.no_grad():
            torch.onnx.export(for_tensorrt(LiveStepFP16(mh, idx)).to(dev), tuple(ins), str(onnx_p), input_names=NAMES_IN,
                              output_names=NAMES_OUT, opset_version=17, dynamo=False, do_constant_folding=True)
        b, info = build_engine(onnx_p, strongly_typed=True, tf32=False)
        assert b is not None, info
        e16 = ENGINES / f"game_b2_live_{style}_fp16.engine"; e16.write_bytes(bytes(b))
        e32 = ENGINES / f"game_b2_live_{style}_fp32_strict.engine"
        t16, t32 = TRT(e16), TRT(e32)
        dps, mads, worst = [], [], 0.0
        for sid in sp["test_sessions"]:
            s = StructuralSession(ROOT / "data_pipeline/recorder" / sid)
            keep = [f for f in s.frame_ids() if str(f) not in ex.get(sid, {})]
            ds = TemporalFrameDataset(s, target_dir_for(sid, style), keep)
            for i in range(1, len(ds), 30):
                bt = ds[i]
                if not bt["has_history"]:
                    continue
                with torch.no_grad():
                    fp, _ = m.encode(bt["structure_prev"][None].to(dev))
                d = {"x_t": bt["structure_curr"][None], "x_prev": bt["structure_prev"][None], "feat_prev": fp,
                     "motion": bt["motion_curr"][None], "no_prev": torch.zeros(1, 1, 288, 512),
                     "motion_valid": torch.ones(1, 1, 1, 1)}
                y = bt["target_curr"][None].to(dev)
                a, _ = t32(d); h, _ = t16(d)
                a, h = a.clamp(0, 1), h.clamp(0, 1)
                dps.append(abs(float(psnr(h, y)) - float(psnr(a, y))))
                mads.append(float((h - a).abs().mean())); worst = max(worst, float((h - a).abs().max()))
        b16, _ = bench(e16); b32, _ = bench(e32)
        ok = float(np.mean(dps)) < TOL_DPSNR and float(np.mean(mads)) < TOL_MAD
        passed &= ok
        rep["styles"][style] = {"n_pairs": len(dps), "mean_abs_dpsnr_db": float(np.mean(dps)),
                                "max_abs_dpsnr_db": float(np.max(dps)), "mean_abs_pixel_diff": float(np.mean(mads)),
                                "max_abs_pixel_diff": worst, "pass": ok, "bench_fp16": b16, "bench_fp32_strict": b32,
                                "engine_fp16": str(e16.relative_to(ROOT))}
        print(style, json.dumps(rep["styles"][style]), flush=True)
    rep["fp16_tolerance_pass"] = bool(passed)
    (ROOT / "eval/reports/m10_fp16_live.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("FP16 tolerance:", "PASS" if passed else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
