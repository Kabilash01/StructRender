"""ONNX -> TensorRT engine builder — KB §4 *TensorRT Builder*.

Reports, never repairs: a build failure returns the parser's own diagnostics so
the M5 export probe can attribute a blocker to a specific operator instead of to
"attention". Batch size 1 and fixed shapes only (§4 Target, §12 mitigation), so
no optimization profile is needed.

§4 also requires the final engine to be built on the target RTX 5060 laptop —
cloud-built engines are not assumed portable — so this module has no remote path.
"""

from __future__ import annotations

from pathlib import Path


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


def tensorrt_version() -> str | None:
    try:
        import tensorrt
    except Exception:
        return None
    return tensorrt.__version__


def build_engine(
    onnx_path: str | Path,
    strongly_typed: bool = False,
    tf32: bool = True,
    workspace_bytes: int = 1 << 30,
):
    """Build a serialized engine. Returns ``(engine_bytes_or_None, info_dict)``.

    ``info_dict`` always carries ``ok`` and, on failure, ``errors``: the list of
    parser errors, which is what names the unsupported operator.

    ``strongly_typed`` builds the network with
    ``NetworkDefinitionCreationFlag.STRONGLY_TYPED``, which takes every layer's
    precision from the ONNX graph's own dtypes. **This is the only way to get an
    FP16 engine on TensorRT 11**: the ``BuilderFlag.FP16`` / ``INT8`` / ``BF16``
    flags that TensorRT <= 10 used were removed, so §4's "primary precision:
    FP16" now means "export an FP16 ONNX graph", not "set a builder flag".

    ``tf32`` is TensorRT's default and is what an FP32 engine actually runs on an
    Ampere-or-later GPU; it costs ~1e-4 of accuracy, which is enough to make a
    numerical-parity check measure the precision mode rather than the operator.
    Pass ``tf32=False`` when the question is "is this operator implemented
    correctly", and leave it on when the question is "how fast does it run".
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        return None, {"ok": False, "stage": "parse", "errors": errors}

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if not tf32 and not strongly_typed:
        config.clear_flag(trt.BuilderFlag.TF32)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None, {"ok": False, "stage": "build", "errors": ["build_serialized_network returned None"]}
    return bytes(serialized), {
        "ok": True,
        "stage": "build",
        "errors": [],
        "strongly_typed": strongly_typed,
        "tf32": tf32 and not strongly_typed,
        "n_layers": network.num_layers,
        "engine_bytes": len(bytes(serialized)),
    }
