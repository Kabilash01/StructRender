"""M4 exit criterion 4 — "export works", and KB §12's first real operator check.

The M4 architecture is low risk; the point of proving PyTorch -> ONNX ->
onnxruntime end to end **now** is that M5+ sends custom attention/masking down
this exact path. Fixed/bounded shapes per §12's mitigation.

Skips cleanly (never a false PASS) when onnx/onnxruntime are absent.

Writes its report to a tmp path; pass ``STRUCTRENDER_REFRESH_REPORTS=1`` to
refresh the committed ``eval/reports/m4_onnx_export.json`` instead.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest
import torch

from inference.export.export_onnx import export_onnx
from inference.export.validate_onnx import check_parity, graph_summary, onnxruntime_available
from renderer.baselines.unet import UNetBaseline, parameter_count
from renderer.training.train_baseline import objective_string, ROOT

pytestmark = pytest.mark.skipif(
    not onnxruntime_available(), reason="onnxruntime not installed (pip install '.[export]')"
)


@pytest.fixture(scope="module")
def exported(cfg, session, tmp_path_factory, report_sink):
    e = cfg["export"]
    torch.manual_seed(cfg["train"]["seed"])
    model = UNetBaseline(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).eval()
    ckpt = Path(ROOT) / cfg["train"]["checkpoint_dir"] / "best.pt"
    if ckpt.is_file():  # prefer the trained weights when they exist
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    path = tmp_path_factory.mktemp("onnx") / "m4_unet_baseline.onnx"
    export_onnx(model, path, tuple(e["input_shape"]), opset=int(e["opset"]))
    summary = graph_summary(path)
    parity = check_parity(model, path, tuple(e["input_shape"]))
    report = {
        # KB Decision 42 self-identification. `trained_weights: true` is
        # meaningless without naming WHICH checkpoint, and an export report that
        # cannot say what it exported is exactly how a stale artifact survives.
        "checkpoint": str(ckpt) if ckpt.is_file() else None,
        # Resolve the ACTUAL session, never a sentinel: "discovered-newest"
        # records that the report did not look, which cannot distinguish a
        # current-session export from a retired-scene one — the exact failure
        # KB Decision 42 exists to close.
        "session_id": session.session_id,
        "style_id": cfg["data"]["style_id"],
        "objective": objective_string(cfg),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "parameters": parameter_count(model),
        "trained_weights": ckpt.is_file(),
        "graph": summary,
        "parity": parity,
        "tolerance_max_abs": e["parity_tolerance_max_abs"],
    }
    out = report_sink("m4_onnx_export.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def test_opset_is_the_configured_one(cfg, exported):
    assert exported["graph"]["opset"] == int(cfg["export"]["opset"])


def test_shapes_are_fixed_not_dynamic(cfg, exported):
    """§12 mitigation: 'prefer fixed/bounded shapes'. Every dim must be a literal
    int — a dim_param (symbolic) would mean a dynamic axis slipped in."""
    e = cfg["export"]
    (in_shape,) = exported["graph"]["inputs"].values()
    (out_shape,) = exported["graph"]["outputs"].values()
    assert in_shape == e["input_shape"] == [1, 7, 288, 512]
    assert out_shape == e["output_shape"] == [1, 3, 288, 512]
    assert all(isinstance(d, int) for d in in_shape + out_shape)


def test_only_onnx_safe_operators_are_used(exported):
    """§12: 'use ONNX-safe primitives'. The allow-list is explicit so that a new
    operator appearing in M5's graph is a visible diff, not a silent surprise.

    ``Shape``/``Reshape``/``InstanceNormalization``/``Mul``/``Add`` are not in
    the model source — they are how the exporter decomposes ``nn.GroupNorm``
    (see :func:`test_groupnorm_decomposition_is_recorded`).
    """
    allowed = {
        "Conv", "Relu", "MaxPool", "Concat", "Resize", "Sigmoid",
        "InstanceNormalization", "GroupNormalization", "Shape", "Reshape",
        "Mul", "Add", "Constant", "Identity",
    }
    used = set(exported["graph"]["op_types"])
    assert used <= allowed, f"unexpected operators in the exported graph: {sorted(used - allowed)}"


def test_groupnorm_decomposition_is_recorded(exported):
    """The one operator finding worth carrying to M5/M10.

    The TorchScript ONNX exporter never emits the native ``GroupNormalization``
    op — at opset 17, 18 and 20 alike it lowers every ``nn.GroupNorm`` to
    ``Shape -> Reshape -> InstanceNormalization -> Reshape -> Mul -> Add``. That
    is TensorRT-supported, so nothing needed working *around*, but it means the
    graph carries a ``Shape`` node per norm layer; if a future export ever needs
    a shape-free graph, the norm layer is where to look.
    """
    ops = exported["graph"]["op_types"]
    assert "GroupNormalization" not in ops
    assert ops["InstanceNormalization"] == ops["Shape"]  # exactly one per GroupNorm


def test_onnxruntime_output_matches_pytorch(exported):
    tol = exported["tolerance_max_abs"]
    got = exported["parity"]["max_abs_diff"]
    assert got <= tol, f"max |onnx - torch| = {got:.3e} > {tol}"


def test_runtime_tensor_contract(exported):
    assert exported["parity"]["onnx_input_shape"] == [1, 7, 288, 512]
    assert exported["parity"]["onnx_output_shape"] == [1, 3, 288, 512]
