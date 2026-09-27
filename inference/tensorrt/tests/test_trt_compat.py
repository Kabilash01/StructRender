"""Decision 84: the GroupNorm rewrite is exact and removes InstanceNormalization."""

import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from inference.export.export_onnx import export_onnx
from inference.export.trt_compat import ExplicitGroupNorm, for_tensorrt


def test_explicit_groupnorm_matches_nn_groupnorm():
    torch.manual_seed(0)
    gn = nn.GroupNorm(8, 32)
    gn.weight.data.normal_(); gn.bias.data.normal_()
    x = torch.randn(2, 32, 9, 16)
    assert torch.allclose(gn(x), ExplicitGroupNorm(gn)(x), atol=1e-5)


def test_rewrite_leaves_the_original_model_untouched():
    m = nn.Sequential(nn.Conv2d(3, 16, 3), nn.GroupNorm(8, 16))
    t = for_tensorrt(m)
    assert isinstance(m[1], nn.GroupNorm) and isinstance(t[1], ExplicitGroupNorm)


def test_exported_graph_has_no_instance_normalization():
    m = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.GroupNorm(8, 16), nn.ReLU(),
                      nn.Conv2d(16, 3, 1), nn.Sigmoid())
    p = export_onnx(for_tensorrt(m), Path(tempfile.mkdtemp()) / "m.onnx", input_shape=(1, 7, 32, 32))
    ops = {n.op_type for n in onnx.load(str(p)).graph.node}
    assert "InstanceNormalization" not in ops
    with torch.no_grad():
        x = torch.rand(1, 7, 32, 32)
        assert torch.allclose(m(x), for_tensorrt(m)(x), atol=1e-5)
