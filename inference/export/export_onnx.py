"""PyTorch -> ONNX export for the M4 baseline (KB §4 ONNX Export Module).

KB §12 *ONNX/TensorRT Operator Compatibility* mitigation, applied from M4 so the
path is proven before M5 sends much higher-risk custom attention/masking down it:

- **fixed shapes only** (``dynamic_axes`` is not used) — the runtime tensor
  contract is a single resolution;
- ONNX-safe primitives only (Conv, GroupNorm-as-primitives, Relu, MaxPool,
  Resize/nearest, Concat, Sigmoid);
- the exported graph is FP32; FP16/FP8 is a TensorRT builder concern
  (Decision 18).
"""

from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(
    model: torch.nn.Module,
    out_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 7, 288, 512),
    opset: int = 17,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().to("cpu").float()
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["structure"],
        output_names=["rgb"],
        opset_version=opset,
        dynamo=False,  # legacy tracing exporter: fixed shapes, no dynamic-shape graph
        do_constant_folding=True,
    )
    return out_path
