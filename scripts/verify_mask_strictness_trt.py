"""Does the STRICT entity mask still reach TensorRT? — KB Decision 52 follow-up.

Decision 52 replaced the additive ``-100`` penalty with masking to ``-inf``,
because a finite penalty only discourages an illegal key and leaks one entity's
memory into another's pixels once logits grow. That changed the exported graph
(``Where``/``masked_fill`` instead of an ``Add``), and Decision 52 recorded the
TensorRT behaviour as UNVERIFIED. This verifies it.

Writes ``eval/reports/m7_mask_strictness_trt.json`` — a NEW artifact. The
historical probe ``eval/reports/m5_export_probe.json`` describes the OLD additive
form and is deliberately left untouched.

    python -m scripts.verify_mask_strictness_trt
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from inference.tensorrt.benchmark.run_engine import run_engine
from inference.tensorrt.builder.build_engine import build_engine, tensorrt_available, tensorrt_version
from renderer.structformer.masked_entity_attention import EntityMaskedCrossAttention
from renderer.training.train_baseline import ROOT

DIM, HEADS, K_MAX, H, W, OPSET = 32, 1, 8, 18, 32, 17
TOLERANCE = 1e-3
SEED = 20260910


class Probe(nn.Module):
    def __init__(self, impl: str):
        super().__init__()
        self.xattn = EntityMaskedCrossAttention(DIM, DIM, HEADS, K_MAX, impl)

    def forward(self, x, entity_tokens, entity_mask):
        return x + self.xattn(x, entity_tokens, entity_mask)


def _inputs():
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, DIM, H, W, generator=g)
    tokens = torch.randn(1, K_MAX, DIM, generator=g)
    mask = torch.zeros(1, K_MAX, H, W)
    # Disjoint rectangles + a genuinely unowned column + padded slots that are
    # visible nowhere. A mask of all ones would let a broken mask op pass.
    mask[0, 0, :, : W // 3] = 1.0
    mask[0, 1, : H // 2, W // 3 : 2 * W // 3] = 1.0
    mask[0, 2, H // 2 :, W // 3 : 2 * W // 3] = 1.0
    return x, tokens, mask


def main() -> int:
    out_path = Path(ROOT) / "eval" / "reports" / "m7_mask_strictness_trt.json"
    report = {
        "purpose": ("TensorRT verification of the STRICT (-inf) entity mask introduced by KB "
                    "Decision 52. The historical probe eval/reports/m5_export_probe.json "
                    "describes the previous additive form and is not modified."),
        "masking": "strict: illegal keys masked to -inf; fully-masked rows uniform then gated to 0.0",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "tensorrt_version": tensorrt_version(),
        "opset": OPSET, "tolerance_max_abs": TOLERANCE,
        "shapes": {"x": [1, DIM, H, W], "entity_tokens": [1, K_MAX, DIM],
                   "entity_mask": [1, K_MAX, H, W]},
        "k_max_probe": K_MAX,
        "variants": {},
    }
    if not (tensorrt_available() and torch.cuda.is_available()):
        report["skipped"] = "tensorrt or CUDA unavailable"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print("[verify] skipped — no TensorRT/CUDA")
        return 0

    x, tokens, mask = _inputs()
    for impl in ("manual", "sdpa"):
        torch.manual_seed(SEED)
        m = Probe(impl).eval()
        with torch.no_grad():
            ref = m(x, tokens, mask).numpy()
        v: dict = {"attn_impl": impl,
                   "torch_output_finite": bool(np.isfinite(ref).all())}

        onnx_path = Path(f"/tmp/mask_strict_{impl}.onnx")
        torch.onnx.export(m, (x, tokens, mask), str(onnx_path),
                          input_names=["x", "entity_tokens", "entity_mask"],
                          output_names=["out"], opset_version=OPSET,
                          dynamo=False, do_constant_folding=True)
        import onnx as _onnx
        g = _onnx.load(str(onnx_path)).graph
        census: dict = {}
        for n in g.node:
            census[n.op_type] = census.get(n.op_type, 0) + 1
        def dims(vi):
            return [d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in vi.type.tensor_type.shape.dim]
        v["onnx_export_ok"] = True
        v["op_types"] = census
        v["shapes_are_static"] = all(
            isinstance(d, int) and d > 0
            for vi in list(g.input) + list(g.output) for d in dims(vi))

        eng, info = build_engine(onnx_path, strongly_typed=False, tf32=False,
                                 workspace_bytes=1 << 29)
        v["trt_fp32"] = {"build_ok": eng is not None,
                         "errors": info.get("errors", []),
                         "n_layers": info.get("n_layers")}
        if eng is not None:
            r = run_engine(eng, {"x": x, "entity_tokens": tokens, "entity_mask": mask},
                           warmup=5, iters=20)
            got = list(r["outputs"].values())[0].float().cpu().numpy()
            d = float(np.abs(got - ref).max())
            v["trt_fp32"].update({
                "run_ok": True, "max_abs_diff": d,
                "within_tolerance": bool(np.isfinite(d) and d <= TOLERANCE),
                "output_finite": bool(np.isfinite(got).all()),
                "latency_ms_mean": r["latency_ms_mean"]})
        report["variants"][impl] = v

    ok = all(v.get("trt_fp32", {}).get("within_tolerance") and v["trt_fp32"].get("output_finite")
             for v in report["variants"].values())
    report["verdict"] = "PASS" if ok else "FAIL"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[verify] wrote {out_path}")
    for k, v in report["variants"].items():
        t = v.get("trt_fp32", {})
        print(f"  {k:7s} onnx={v['onnx_export_ok']} static={v['shapes_are_static']} "
              f"trt_build={t.get('build_ok')} max|diff|={t.get('max_abs_diff')} "
              f"finite={t.get('output_finite')}")
    print(f"[verify] {report['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
