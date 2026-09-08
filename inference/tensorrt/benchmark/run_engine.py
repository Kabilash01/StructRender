"""Run a serialized TensorRT engine and time it — KB §4 *TensorRT Builder* /
§12 *Performance Bottlenecks*.

Uses torch CUDA tensors as the I/O buffers so there is no pycuda dependency and
the comparison against PyTorch stays in one process. Batch 1, fixed shapes.
"""

from __future__ import annotations

import torch


def run_engine(engine_bytes: bytes, inputs: dict, warmup: int = 10, iters: int = 50) -> dict:
    """Execute ``engine_bytes`` on ``inputs`` (name -> CPU float32 tensor).

    Returns ``{"outputs": {name: cpu tensor}, "latency_ms_mean": .., "latency_ms_p50": ..}``.
    Latency is measured with CUDA events around a single ``execute_async_v3``.
    """
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None")
    context = engine.create_execution_context()

    dtype_map = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}
    buffers: dict[str, torch.Tensor] = {}
    outputs: list[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = dtype_map[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name not in inputs:
                raise KeyError(f"engine input {name!r} not supplied; got {sorted(inputs)}")
            buffers[name] = inputs[name].to("cuda", dtype).contiguous()
        else:
            buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
            outputs.append(name)
        context.set_tensor_address(name, buffers[name].data_ptr())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.cuda.stream(stream):
        for _ in range(iters):
            start.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            end.record(stream)
            end.synchronize()
            times.append(start.elapsed_time(end))
    stream.synchronize()

    times.sort()
    return {
        "outputs": {n: buffers[n].float().cpu() for n in outputs},
        "latency_ms_mean": sum(times) / len(times),
        "latency_ms_p50": times[len(times) // 2],
        "latency_ms_min": times[0],
    }
