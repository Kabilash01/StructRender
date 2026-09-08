"""M5 step 2 — window-attention ONNX/TensorRT export probe.

KB §12 flags exactly one operator risk for this project: "custom
attention/masking operation exports poorly", mitigated by "use ONNX-safe
primitives; perform TensorRT export in Week 5; prefer fixed/bounded shapes".
M5 is the first milestone that sends attention down the ONNX -> TensorRT path,
so this probe runs **before** the architecture is trained: if an operator blocks
TensorRT it changes the design, and finding that out after a training run wastes
the compute budget KB Decision 20 protects.

What this is and is not:

- It is **op-identical** to the shipped renderer. Every variant is built from the
  real modules in :mod:`renderer.structformer`, instantiated at minimum capacity
  (embed dim 32, 1 head, 1-2 blocks). A throwaway toy would prove something about
  code that will never run.
- It is **not** a capacity, quality or budget measurement. Weights are random and
  untrained; widths are minimal. The latency numbers say "this graph runs and
  costs this much at this width" — they are **not** evidence about §12's
  15-20 ms StructFormer budget, which depends on the trained width.

Each KB §12 hazard is a separate variant so that a blocker is attributable to an
operator, not to "attention":

1. window partition / reverse (reshape + permute);
2. the attention form itself — ``F.scaled_dot_product_attention`` vs an explicit
   matmul/softmax/matmul, exported separately because they lower differently;
3. relative position bias — ``Gather`` on a learned table via a constant index;
4. an additive attention mask added before softmax;
5. cyclic shift — ``torch.roll`` and a slice+concat equivalent;
6. M7's entity-masked cross-attention (probe only, not in the M5 model).

Usage::

    /home/kabilash01/miniconda3/envs/dl/bin/python scripts/probe_m5_export.py \
        --out eval/reports/m5_export_probe.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderer.structformer.masked_entity_attention import EntityMaskedCrossAttention  # noqa: E402
from renderer.structformer.model import StructFormer, parameter_count  # noqa: E402
from renderer.structformer.window_attention import (  # noqa: E402
    RelativePositionBias,
    WindowAttention,
    WindowTransformerBlock,
    build_shift_attention_mask,
    window_partition,
    window_reverse,
)

OPSET = 17  # same as M4 (KB Decision 35), so the graphs are comparable
TOLERANCE_MAX_ABS = 1e-3  # same as M4's parity tolerance
SEED = 20260908

# Probe geometry. 288x512 / stem stride 4 -> 72x128; window 8 divides both.
DIM = 32
HEADS = 1
WS = 8
RES = (72, 128)
K_MAX = 8  # probe-only cap; §5's runtime contract recommends 32 then 64
ENTITY_RES = (36, 64)


# --------------------------------------------------------------------------
# Probe wrappers. These exist only to give each hazard its own graph; all the
# arithmetic inside them comes from renderer.structformer.
# --------------------------------------------------------------------------


class PartitionReverseOnly(nn.Module):
    """Hazard 1 in isolation: the reshape+permute round trip, no attention."""

    def __init__(self, window_size: int = WS) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1)
        win = window_partition(y, self.window_size)
        y = window_reverse(win, self.window_size, h, w)
        return y.permute(0, 3, 1, 2).contiguous()


class AttentionOnly(nn.Module):
    """Hazards 2-4: one :class:`WindowAttention` between partition and reverse.

    ``mask`` is registered as a constant buffer when requested, so the additive
    pre-softmax mask op is present in the graph even though M5's own mask is
    trivial — M7's will not be, and §12 wants that blocker found now.
    """

    def __init__(self, use_rel_pos_bias: bool, use_mask: bool, attn_impl: str) -> None:
        super().__init__()
        self.window_size = WS
        self.attn = WindowAttention(DIM, WS, HEADS, use_rel_pos_bias, attn_impl)
        if use_mask:
            self.register_buffer(
                "mask", build_shift_attention_mask(RES[0], RES[1], WS, WS // 2), persistent=False
            )
        else:
            self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1)
        win = window_partition(y, self.window_size)
        win = self.attn(win, self.mask)
        y = window_reverse(win, self.window_size, h, w)
        return y.permute(0, 3, 1, 2).contiguous()


class RelPosBiasOnly(nn.Module):
    """Hazard 3 in isolation: the ``Gather`` on the learned table, nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = RelativePositionBias(WS, HEADS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias().unsqueeze(0)


class EntityCrossAttentionProbe(nn.Module):
    """Hazard 6: M7's entity-masked cross-attention. Probe only, not in Model B."""

    def __init__(self, attn_impl: str) -> None:
        super().__init__()
        self.xattn = EntityMaskedCrossAttention(DIM, DIM, HEADS, K_MAX, attn_impl)

    def forward(
        self, x: torch.Tensor, entity_tokens: torch.Tensor, entity_mask: torch.Tensor
    ) -> torch.Tensor:
        return x + self.xattn(x, entity_tokens, entity_mask)


# --------------------------------------------------------------------------
# Deterministic inputs
# --------------------------------------------------------------------------


def make_inputs(spec: dict[str, tuple], seed: int = SEED) -> dict[str, torch.Tensor]:
    """Random-but-deterministic inputs. ``entity_mask`` is built as real, disjoint
    rectangles rather than noise — a mask that is 1 everywhere would let a broken
    mask op pass."""
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for name, shape in spec.items():
        if name == "entity_mask":
            b, k, h, w = shape
            m = torch.zeros(shape)
            for i in range(k - 1):  # last slot stays empty = an unused padded slot
                r0 = (i * h) // k
                m[:, i, r0 : r0 + max(1, h // (k + 2)), (i * w) // k : (i * w) // k + w // (k + 2)] = 1.0
            out[name] = m
        else:
            out[name] = torch.randn(shape, generator=g)
    return out


# --------------------------------------------------------------------------
# ONNX / TensorRT plumbing
# --------------------------------------------------------------------------


def export(model: nn.Module, path: Path, inputs: dict[str, torch.Tensor]) -> None:
    """Same exporter call as M4 (``dynamo=False``, constant folding, fixed shapes).

    M4's :func:`inference.export.export_onnx.export_onnx` is single-input by
    construction; several probe variants take three. Rather than modify a
    committed M4 artefact this mirrors its settings exactly.
    """
    torch.onnx.export(
        model.eval(),
        tuple(inputs.values()),
        str(path),
        input_names=list(inputs),
        output_names=["out"],
        opset_version=OPSET,
        dynamo=False,
        do_constant_folding=True,
    )


def graph_facts(path: Path) -> dict:
    """Operator census + the two structural properties §12 asks us to verify."""
    import onnx

    m = onnx.load(str(path))
    ops: dict[str, int] = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    def shape_of(vi):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in vi.type.tensor_type.shape.dim]

    initializers = {i.name for i in m.graph.initializer}
    constants = {o for n in m.graph.node if n.op_type == "Constant" for o in n.output}
    gathers = [n for n in m.graph.node if n.op_type == "Gather"]
    # Hazard 3: the relative-position index must be baked in, not computed.
    gather_indices_constant = [
        (n.input[1] in initializers or n.input[1] in constants) for n in gathers
    ]

    in_shapes = {vi.name: shape_of(vi) for vi in m.graph.input}
    out_shapes = {vi.name: shape_of(vi) for vi in m.graph.output}
    all_dims = [d for s in list(in_shapes.values()) + list(out_shapes.values()) for d in s]
    return {
        "opset": max(i.version for i in m.opset_import if i.domain in ("", "ai.onnx")),
        "inputs": in_shapes,
        "outputs": out_shapes,
        "shapes_are_static": all(isinstance(d, int) for d in all_dims),
        "op_types": dict(sorted(ops.items())),
        "n_nodes": len(m.graph.node),
        "n_gather_nodes": len(gathers),
        "gather_indices_all_constant": all(gather_indices_constant) if gathers else None,
        "size_bytes": path.stat().st_size,
    }


def ort_parity(path: Path, inputs: dict, ref: torch.Tensor) -> dict:
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        return {"available": False, "error": str(e)}
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: v.numpy() for k, v in inputs.items()})[0]
    return {
        "available": True,
        "max_abs_diff": float((torch.from_numpy(got) - ref).abs().max()),
        "version": ort.__version__,
    }


def trt_trial(path: Path, inputs: dict, ref: torch.Tensor, strongly_typed: bool,
              tf32: bool) -> dict:
    from inference.tensorrt.builder.build_engine import build_engine
    from inference.tensorrt.benchmark.run_engine import run_engine

    engine, info = build_engine(path, strongly_typed=strongly_typed, tf32=tf32)
    result = {
        "build_ok": info["ok"],
        "strongly_typed": strongly_typed,
        "tf32": tf32 and not strongly_typed,
        "errors": info["errors"],
    }
    if not info["ok"]:
        return result
    result["n_layers"] = info["n_layers"]
    result["engine_bytes"] = info["engine_bytes"]
    try:
        run = run_engine(engine, inputs)
    except Exception as e:
        result["run_ok"] = False
        result["errors"] = [f"{type(e).__name__}: {e}"]
        return result
    got = next(iter(run["outputs"].values()))
    result["run_ok"] = True
    result["max_abs_diff"] = float((got - ref).abs().max())
    result["latency_ms_mean"] = round(run["latency_ms_mean"], 4)
    result["latency_ms_p50"] = round(run["latency_ms_p50"], 4)
    result["latency_ms_min"] = round(run["latency_ms_min"], 4)
    return result


def run_variant(name: str, meta: dict, model: nn.Module, spec: dict, workdir: Path,
                m4_ops: set[str], trt: bool) -> dict:
    torch.manual_seed(SEED)
    model = model.eval()
    inputs = make_inputs(spec)
    with torch.no_grad():
        ref = model(*inputs.values())

    entry = dict(meta)
    entry["parameters"] = parameter_count(model)
    entry["output_shape"] = list(ref.shape)
    path = workdir / f"{name}.onnx"
    try:
        export(model, path, inputs)
        entry["onnx_export_ok"] = True
    except Exception as e:
        entry["onnx_export_ok"] = False
        entry["onnx_error"] = f"{type(e).__name__}: {e}"
        return entry

    facts = graph_facts(path)
    entry["graph"] = facts
    entry["new_ops_vs_m4"] = sorted(set(facts["op_types"]) - m4_ops)
    entry["onnxruntime"] = ort_parity(path, inputs, ref)
    if trt:
        # tf32 off: isolates "is the operator correct" from "which precision mode
        # did TensorRT pick". TF32 is the default and is what an FP32 engine
        # really runs on this GPU, so it stays on for the FP16 comparison.
        entry["trt"] = {"fp32": trt_trial(path, inputs, ref, strongly_typed=False, tf32=False)}
        # FP16 needs a half-precision ONNX graph on TensorRT 11 (see
        # build_engine's docstring). Traced on CUDA because several fp16 kernels
        # (LayerNorm, GELU) have no CPU implementation.
        fp16_path = workdir / f"{name}_fp16.onnx"
        try:
            half = model.half().cuda()
            half_inputs = {k: v.half().cuda() for k, v in inputs.items()}
            with torch.no_grad():
                export(half, fp16_path, half_inputs)
            entry["trt"]["fp16"] = trt_trial(
                fp16_path, {k: v.cpu() for k, v in half_inputs.items()}, ref,
                strongly_typed=True, tf32=True,
            )
        except Exception as e:
            entry["trt"]["fp16"] = {"build_ok": False, "strongly_typed": True,
                                    "errors": [f"fp16 export failed: {type(e).__name__}: {e}"]}
        finally:
            model.float().cpu()
    return entry


def build_variants() -> list[tuple[str, dict, nn.Module, dict]]:
    feat = {"x": (1, DIM, RES[0], RES[1])}
    ent = {
        "x": (1, DIM, ENTITY_RES[0], ENTITY_RES[1]),
        "entity_tokens": (1, K_MAX, DIM),
        "entity_mask": (1, K_MAX, ENTITY_RES[0], ENTITY_RES[1]),
    }
    e2e = {"structure": (1, 7, 288, 512), "style_token": (1, 16)}
    v: list[tuple[str, dict, nn.Module, dict]] = []

    v.append((
        "h1_window_partition_reverse",
        {"hazard": "1 window partition/reverse", "attn_impl": None, "shift_impl": None},
        PartitionReverseOnly(), feat,
    ))
    v.append((
        "h3_rel_pos_bias_only",
        {"hazard": "3 relative position bias (gather on learned table)",
         "attn_impl": None, "shift_impl": None},
        RelPosBiasOnly(), {"x": (1, HEADS, WS * WS, WS * WS)},
    ))
    for impl in ("manual", "sdpa"):
        v.append((
            f"h2_attn_{impl}",
            {"hazard": "2 attention form, no bias, no mask", "attn_impl": impl, "shift_impl": None},
            AttentionOnly(False, False, impl), feat,
        ))
        v.append((
            f"h23_attn_{impl}_relpos",
            {"hazard": "2+3 attention with relative position bias",
             "attn_impl": impl, "shift_impl": None},
            AttentionOnly(True, False, impl), feat,
        ))
        v.append((
            f"h234_attn_{impl}_relpos_mask",
            {"hazard": "2+3+4 attention with bias and additive pre-softmax mask",
             "attn_impl": impl, "shift_impl": None},
            AttentionOnly(True, True, impl), feat,
        ))
        v.append((
            f"h5_block_{impl}_noshift",
            {"hazard": "5 control: full block, shift_size=0 (no roll)",
             "attn_impl": impl, "shift_impl": "n/a"},
            WindowTransformerBlock(DIM, RES, HEADS, WS, 0, attn_impl=impl), feat,
        ))
        for shift_impl in ("roll", "slice"):
            v.append((
                f"h5_block_{impl}_shift_{shift_impl}",
                {"hazard": f"5 shifted window via {shift_impl}",
                 "attn_impl": impl, "shift_impl": shift_impl},
                WindowTransformerBlock(DIM, RES, HEADS, WS, WS // 2, attn_impl=impl,
                                       shift_impl=shift_impl), feat,
            ))
        v.append((
            f"h6_entity_masked_xattn_{impl}",
            {"hazard": "6 M7 entity-masked cross-attention (probe only, K_max=8)",
             "attn_impl": impl, "shift_impl": None},
            EntityCrossAttentionProbe(impl), ent,
        ))
        v.append((
            f"e2e_structformer_{impl}_shift",
            {"hazard": "end-to-end fixed-shape path, depth 2 (shifted blocks present)",
             "attn_impl": impl, "shift_impl": "roll"},
            StructFormer(depths=(2, 2, 2), attn_impl=impl), e2e,
        ))
    v.append((
        "e2e_structformer_manual_noshift",
        {"hazard": "end-to-end control, depth 1 (no shifted block, no roll)",
         "attn_impl": "manual", "shift_impl": "n/a"},
        StructFormer(depths=(1, 1, 1), attn_impl="manual"), e2e,
    ))
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports" / "m5_export_probe.json"))
    ap.add_argument("--workdir", default=None, help="where to keep the .onnx files")
    ap.add_argument("--no-trt", action="store_true", help="ONNX only (no GPU available)")
    ap.add_argument("--only", default=None, help="substring filter on variant name")
    args = ap.parse_args()

    from inference.tensorrt.builder.build_engine import tensorrt_available, tensorrt_version

    trt = tensorrt_available() and torch.cuda.is_available() and not args.no_trt
    # .onnx files are gitignored build artefacts, so they go under checkpoints/,
    # never next to the committed report.
    workdir = Path(args.workdir) if args.workdir else ROOT / "checkpoints" / "m5_export_probe"
    workdir.mkdir(parents=True, exist_ok=True)

    m4_report = ROOT / "eval" / "reports" / "m4_onnx_export.json"
    m4_ops = set(json.loads(m4_report.read_text())["graph"]["op_types"]) if m4_report.is_file() else set()

    variants: dict[str, dict] = {}
    for name, meta, model, spec in build_variants():
        if args.only and args.only not in name:
            continue
        print(f"[probe] {name} ...", flush=True)
        variants[name] = run_variant(name, meta, model, spec, workdir, m4_ops, trt)
        e = variants[name]
        tag = "onnx-FAIL" if not e["onnx_export_ok"] else (
            "ok" if not trt else
            ("trt-FAIL" if not e["trt"]["fp32"].get("run_ok") else
             f"ok  fp32 {e['trt']['fp32']['max_abs_diff']:.2e} "
             f"{e['trt']['fp32']['latency_ms_mean']:.3f} ms")
        )
        print(f"[probe] {name}: {tag}", flush=True)

    import onnx
    import onnxruntime

    all_new = sorted({op for v in variants.values() if v["onnx_export_ok"]
                      for op in v["new_ops_vs_m4"]})
    report = {
        "milestone": 5,
        "step": "2 — window-attention export probe (design-time, pre-architecture)",
        "purpose": "KB §12 'custom attention/masking operation exports poorly' — prove "
                   "each attention hazard reaches TensorRT before the M5 architecture is "
                   "trained, so a blocker changes the design instead of wasting a run.",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "tensorrt_version": tensorrt_version(),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "opset": OPSET,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "trt_probed": trt,
        "trained_weights": False,
        "weights_note": "Random init, no training. This probe measures operator support "
                        "and shape flow, never output quality.",
        "capacity_note": f"embed_dim={DIM}, heads={HEADS}, 1-2 blocks per stage, window "
                         f"{WS}. Minimum capacity, op-identical to the intended model.",
        "latency_note": "Latency is for THIS minimal-width graph on this GPU. It is NOT "
                        "evidence about §12's 15-20 ms StructFormer budget, which depends "
                        "on the trained width. Compare variants against each other only.",
        "numerics_note": "trt.fp32 is built with TF32 disabled so max_abs_diff measures "
                         "operator correctness rather than TensorRT's default precision "
                         "mode; it is gated at tolerance_max_abs. trt.fp16 is the §4 target "
                         "precision — its max_abs_diff is reported against the SAME fp32 "
                         "PyTorch reference and is deliberately NOT gated, because its error "
                         "is precision, not operator, error.",
        "trt11_api_note": "TensorRT 11 removed BuilderFlag.FP16/INT8/BF16. Reduced precision "
                          "now comes only from a STRONGLY_TYPED network whose dtypes are "
                          "those of the ONNX graph, so an FP16 engine requires an FP16 export.",
        "tolerance_max_abs": TOLERANCE_MAX_ABS,
        "k_max_probe": K_MAX,
        "m4_operator_baseline": sorted(m4_ops),
        "new_ops_vs_m4_union": all_new,
        "variants": variants,
    }
    report["summary"] = {
        "n_variants": len(variants),
        "onnx_export_ok": sum(v["onnx_export_ok"] for v in variants.values()),
        "trt_fp32_build_ok": sum(
            bool(v.get("trt", {}).get("fp32", {}).get("build_ok")) for v in variants.values()),
        "trt_fp16_build_ok": sum(
            bool(v.get("trt", {}).get("fp16", {}).get("build_ok")) for v in variants.values()),
        "trt_fp16_run_ok": sum(
            bool(v.get("trt", {}).get("fp16", {}).get("run_ok")) for v in variants.values()),
        "trt_fp32_within_tolerance": sum(
            v.get("trt", {}).get("fp32", {}).get("max_abs_diff", 1e9) <= TOLERANCE_MAX_ABS
            for v in variants.values()),
        "blocked": sorted(n for n, v in variants.items()
                          if not v["onnx_export_ok"]
                          or (trt and not v.get("trt", {}).get("fp32", {}).get("run_ok"))),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
