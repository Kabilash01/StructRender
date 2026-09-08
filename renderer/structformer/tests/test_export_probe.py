"""KB §12 operator compatibility for M5's attention — live checks + the artifact.

Three layers, each skipping cleanly rather than false-passing:

1. **ONNX** (needs ``onnx``): a real :class:`WindowTransformerBlock` exports with
   static shapes, a constant relative-position index, and only operators on the
   M5 allow-list. Extends M4's allow-list discipline — a *new* operator in a
   future graph must show up as a visible diff, exactly as
   ``renderer/training/tests/test_onnx_export.py`` does for M4.
2. **TensorRT** (needs ``tensorrt`` + CUDA): the same block builds an engine,
   runs, and matches PyTorch.
3. **The committed artifact** ``eval/reports/m5_export_probe.json``: asserted to
   still say what it is quoted as saying. KB Decision 44's finding was that
   nothing checked the *committed* report, so a GPU-less machine could pass a
   whole suite over a stale or failing artifact.

These tests never write to ``eval/reports/``; the report is produced only by
``scripts/probe_m5_export.py`` (KB Decision 35, "test runs must not overwrite
committed evidence").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from renderer.structformer.tests.conftest import ROOT
from renderer.structformer.window_attention import WindowTransformerBlock

OPSET = 17
REPORT = ROOT / "eval" / "reports" / "m5_export_probe.json"

#: Everything M5's graphs are allowed to contain. The first block is M4's
#: allow-list (``eval/reports/m4_onnx_export.json``); the second is what
#: attention adds and is the deliberate, visible diff §12 asks for.
ALLOWED_OPS = {
    # M4 (CNN baseline)
    "Add", "Concat", "Constant", "Conv", "InstanceNormalization", "MaxPool",
    "Mul", "Relu", "Reshape", "Resize", "Shape", "Sigmoid", "Identity",
    # M5 (window attention, LayerNorm, GELU, style FiLM, entity masking)
    "Cast", "ConstantOfShape", "Div", "Erf", "Expand", "Gather", "Gemm",
    "LayerNormalization", "MatMul", "ReduceMax", "Slice", "Softmax", "Sqrt",
    "Sub", "Tile", "Transpose", "Unsqueeze",
}


def _onnx_or_skip():
    pytest.importorskip("onnx", reason="pip install '.[export]'")


def _trt_or_skip():
    pytest.importorskip("tensorrt", reason="TensorRT not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.fixture(scope="module")
def exported_block(tmp_path_factory):
    _onnx_or_skip()
    from inference.export.export_onnx import export_onnx

    torch.manual_seed(0)
    block = WindowTransformerBlock(8, (8, 16), 1, 4, shift_size=2).eval()
    path = tmp_path_factory.mktemp("m5onnx") / "block.onnx"
    export_onnx(block, path, (1, 8, 8, 16), opset=OPSET)
    return block, path


def test_block_exports_with_static_shapes(exported_block):
    import onnx

    _, path = exported_block
    m = onnx.load(str(path))
    dims = [d for vi in list(m.graph.input) + list(m.graph.output)
            for d in vi.type.tensor_type.shape.dim]
    assert all(d.HasField("dim_value") for d in dims), "a symbolic dim reached the graph"


def test_only_allowed_operators_are_used(exported_block):
    import onnx

    _, path = exported_block
    used = {n.op_type for n in onnx.load(str(path)).graph.node}
    assert used <= ALLOWED_OPS, f"unexpected operators: {sorted(used - ALLOWED_OPS)}"


def test_relative_position_index_is_a_graph_constant(exported_block):
    """KB §12 hazard 3: the index must be baked in. A ``Gather`` whose indices
    are computed at runtime means the graph is recomputing ``meshgrid`` every
    frame and carries a larger operator-support surface."""
    import onnx

    _, path = exported_block
    m = onnx.load(str(path))
    constant_names = {i.name for i in m.graph.initializer}
    constant_names |= {o for n in m.graph.node if n.op_type == "Constant" for o in n.output}
    gathers = [n for n in m.graph.node if n.op_type == "Gather"]
    assert gathers, "the relative-position bias should produce at least one Gather"
    assert all(n.input[1] in constant_names for n in gathers)


def test_tensorrt_engine_matches_pytorch(exported_block):
    _trt_or_skip()
    from inference.tensorrt.benchmark.run_engine import run_engine
    from inference.tensorrt.builder.build_engine import build_engine

    block, path = exported_block
    engine, info = build_engine(path, tf32=False)
    assert info["ok"], info["errors"]
    x = torch.randn(1, 8, 8, 16)
    with torch.no_grad():
        ref = block(x)
    got = next(iter(run_engine(engine, {"structure": x}, warmup=3, iters=5)["outputs"].values()))
    assert (got - ref).abs().max() <= 1e-3


# --------------------------------------------------------------------------
# The committed artifact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    if not REPORT.is_file():
        pytest.skip(f"{REPORT} not generated (run scripts/probe_m5_export.py)")
    return json.loads(REPORT.read_text())


def test_report_carries_its_provenance(report):
    """KB Decision 42: a report that cannot say what produced it is how a stale
    artifact survives."""
    for key in ("generated_utc", "torch_version", "tensorrt_version", "opset",
                "onnx_version", "onnxruntime_version", "device", "trained_weights"):
        assert report.get(key) is not None, f"missing provenance field {key}"
    assert report["opset"] == OPSET
    assert report["trained_weights"] is False  # it is an operator probe, not a model


def test_every_probed_hazard_is_covered(report):
    """A probe that quietly stopped exercising a hazard would still report all
    green, so the hazard list itself is asserted."""
    names = set(report["variants"])
    for marker in ("h1_window_partition_reverse", "h3_rel_pos_bias_only",
                   "h234_attn_manual_relpos_mask", "h234_attn_sdpa_relpos_mask",
                   "h5_block_manual_shift_roll", "h5_block_manual_shift_slice",
                   "h6_entity_masked_xattn_manual", "e2e_structformer_manual_shift"):
        assert marker in names, f"hazard variant {marker} missing from the probe"


def test_committed_probe_reports_no_blocked_operator(report):
    assert report["summary"]["blocked"] == []
    for name, v in report["variants"].items():
        assert v["onnx_export_ok"], f"{name}: {v.get('onnx_error')}"
        assert v["graph"]["shapes_are_static"], f"{name}: symbolic dim in the graph"
        assert set(v["graph"]["op_types"]) <= ALLOWED_OPS, (
            f"{name}: {sorted(set(v['graph']['op_types']) - ALLOWED_OPS)}")


def test_committed_probe_engines_built_and_matched_pytorch(report):
    if not report["trt_probed"]:
        pytest.skip("committed report was generated without TensorRT")
    tol = report["tolerance_max_abs"]
    for name, v in report["variants"].items():
        fp32 = v["trt"]["fp32"]
        assert fp32["build_ok"] and fp32["run_ok"], f"{name}: {fp32['errors']}"
        assert fp32["max_abs_diff"] <= tol, f"{name}: {fp32['max_abs_diff']} > {tol}"
        assert v["trt"]["fp16"]["build_ok"], f"{name} fp16: {v['trt']['fp16']['errors']}"


def test_the_end_to_end_graph_keeps_the_m4_tensor_contract(report):
    """Model C must consume and produce exactly what Model A did, or the §11
    ladder compares two different problems."""
    g = report["variants"]["e2e_structformer_manual_shift"]["graph"]
    assert g["inputs"]["structure"] == [1, 7, 288, 512]
    assert g["outputs"]["out"] == [1, 3, 288, 512]
