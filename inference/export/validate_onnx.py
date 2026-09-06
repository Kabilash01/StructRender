"""Load an exported ONNX graph in onnxruntime and check parity against PyTorch.

Reports, never repairs. Returns a dict; the caller decides pass/fail against the
tolerance in ``configs/training/*.json`` -> ``export.parity_tolerance_max_abs``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def graph_summary(onnx_path: str | Path) -> dict:
    import onnx

    m = onnx.load(str(onnx_path))
    ops: dict[str, int] = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    def shape_of(vi) -> list:
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in vi.type.tensor_type.shape.dim]

    return {
        "ir_version": m.ir_version,
        "opset": max(i.version for i in m.opset_import if i.domain in ("", "ai.onnx")),
        "producer": m.producer_name,
        "inputs": {vi.name: shape_of(vi) for vi in m.graph.input},
        "outputs": {vi.name: shape_of(vi) for vi in m.graph.output},
        "op_types": dict(sorted(ops.items())),
        "size_bytes": Path(onnx_path).stat().st_size,
    }


def check_parity(
    model: torch.nn.Module,
    onnx_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 7, 288, 512),
    n_samples: int = 3,
    seed: int = 20260905,
) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    model = model.eval().to("cpu").float()
    rng = np.random.default_rng(seed)

    max_abs, mean_abs = [], []
    for _ in range(n_samples):
        x = rng.standard_normal(input_shape).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {in_name: x})[0]
        if got.shape != ref.shape:
            raise AssertionError(f"onnx output shape {got.shape} != torch {ref.shape}")
        max_abs.append(float(np.abs(got - ref).max()))
        mean_abs.append(float(np.abs(got - ref).mean()))

    return {
        "onnx_input_shape": list(sess.get_inputs()[0].shape),
        "onnx_output_shape": list(sess.get_outputs()[0].shape),
        "n_samples": n_samples,
        "max_abs_diff": max(max_abs),
        "mean_abs_diff": float(np.mean(mean_abs)),
        "onnxruntime_version": ort.__version__,
        "providers": sess.get_providers(),
    }
