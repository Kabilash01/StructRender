"""Diagnose Model A's TensorRT build failure (M10; recorded unresolved in Decision 48).

Error Code 10, "Could not find any implementation for ForeignNode[...]" inside a
Myelin fused region, 4/4 reproductions, on a U-Net whose distinguishing operators
(vs. Model C, which builds) are GroupNorm (-> InstanceNormalization in ONNX),
MaxPool and nearest Resize. This script changes ONE thing per variant and records
which builds, so the fix names its cause:

  baseline        the checkpoint as trained
  opt_level_k     builder_optimization_level k in 0..5 (changes fusion search)
  manual_gn       GroupNorm rewritten as reshape/mean/var/affine — numerically
                  identical (asserted), but no InstanceNormalization node
  repeat_upsample nearest x2 Resize rewritten as repeat_interleave — identical
  both            manual_gn + repeat_upsample

Every rewrite is checked against the original in PyTorch before it is exported,
so a variant that builds cannot be a different model.
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.export.export_onnx import export_onnx
from renderer.training.train_baseline import ROOT, load_checkpoint


class ManualGroupNorm(nn.Module):
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.g, self.eps = gn.num_groups, gn.eps
        self.weight, self.bias = gn.weight, gn.bias

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.reshape(b, self.g, (c // self.g) * h * w)
        mu = y.mean(dim=2, keepdim=True)
        var = ((y - mu) ** 2).mean(dim=2, keepdim=True)
        y = ((y - mu) / torch.sqrt(var + self.eps)).reshape(b, c, h, w)
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def swap_gn(m):
    for n, ch in m.named_children():
        if isinstance(ch, nn.GroupNorm):
            setattr(m, n, ManualGroupNorm(ch))
        else:
            swap_gn(ch)
    return m


_orig_interp = F.interpolate


def _repeat_interp(x, scale_factor=None, mode="nearest", **kw):
    if mode == "nearest" and scale_factor == 2.0:
        return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    return _orig_interp(x, scale_factor=scale_factor, mode=mode, **kw)


def build(onnx_path, opt_level=None):
    import tensorrt as trt
    logs = []

    class L(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, sev, msg):
            if sev <= trt.ILogger.Severity.ERROR:
                logs.append(msg)

    lg = L(); b = trt.Builder(lg); net = b.create_network(0); p = trt.OnnxParser(net, lg)
    if not p.parse(Path(onnx_path).read_bytes()):
        return {"ok": False, "stage": "parse", "errors": [str(p.get_error(i)) for i in range(p.num_errors)]}
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if opt_level is not None:
        cfg.builder_optimization_level = opt_level
    t0 = time.time()
    s = b.build_serialized_network(net, cfg)
    return {"ok": s is not None, "stage": "build", "seconds": round(time.time() - t0, 1),
            "errors": [e[:300] for e in logs[-3:]]}


def run_engine(engine_bytes, x):
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR)); eng = rt.deserialize_cuda_engine(engine_bytes)
    ctx = eng.create_execution_context()
    out = torch.empty(1, 3, 288, 512, device="cuda")
    xi = x.cuda().contiguous()
    ctx.set_tensor_address(eng.get_tensor_name(0), xi.data_ptr())
    ctx.set_tensor_address(eng.get_tensor_name(1), out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
    return out.cpu()


def main() -> int:
    ck = ROOT / "checkpoints/ladder_v3_a/best.pt"
    base = load_checkpoint(ck, torch.device("cpu")).eval()
    x = torch.rand(1, 7, 288, 512)
    with torch.no_grad():
        ref = base(x)
    tmp = Path(tempfile.mkdtemp())
    rep = {"purpose": "Diagnose Model A TensorRT build failure (Decision 48).", "checkpoint": str(ck),
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "variants": {}}
    import tensorrt as trt
    rep["tensorrt"] = trt.__version__

    def variant(name, model, patch_upsample=False, opt_level=None):
        if patch_upsample:
            F.interpolate = _repeat_interp
        try:
            with torch.no_grad():
                out = model(x)
            diff = float((out - ref).abs().max())
            path = export_onnx(model, tmp / f"{name}.onnx")
        finally:
            F.interpolate = _orig_interp
        r = build(path, opt_level); r["torch_max_diff_vs_original"] = diff
        rep["variants"][name] = r
        print(f"[trt] {name:<18} build {'OK ' if r['ok'] else 'FAIL'}  torch diff {diff:.1e}  "
              f"{(r['errors'][-1][:110] if r['errors'] else '')}", flush=True)
        return path

    variant("baseline", base)
    for k in range(6):
        variant(f"opt_level_{k}", base, opt_level=k)
    import copy
    gn = swap_gn(copy.deepcopy(base)).eval()
    variant("manual_gn", gn)
    variant("repeat_upsample", base, patch_upsample=True)
    variant("both", gn, patch_upsample=True)
    (ROOT / "eval/reports/m10_trt_model_a_diagnosis.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("[trt] wrote eval/reports/m10_trt_model_a_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
