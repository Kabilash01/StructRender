"""KB Decision 80: Model B must train through the temporal trainer.

Its ladder-v2 run went through train_baseline, whose data path has no history, so
TemporalUNet.forward(x) never called `fuse` and the fusion received no gradient
for the whole run. The resulting 'inert fusion' was an artifact of the trainer,
not a property of the mechanism.
"""

import json

import pytest
import torch

from renderer.training import train_temporal as T
from renderer.training.train_baseline import ROOT


def _cfg():
    return json.load(open(ROOT / "configs/ladder_v3/model_b.json"))


def test_train_baseline_refuses_temporal_unet(monkeypatch):
    from renderer.training import train_baseline as B
    monkeypatch.setattr("sys.argv", ["x", "--config", "configs/ladder_v3/model_b.json", "--mode", "train"])
    with pytest.raises(SystemExit) as e:
        B.main()
    assert "train_temporal" in str(e.value)


def test_train_temporal_builds_model_b():
    from renderer.temporal.temporal_unet import TemporalUNet
    assert isinstance(T.build_model(_cfg()), TemporalUNet)


def test_model_b_fusion_gets_gradient_with_history_and_none_without():
    torch.manual_seed(0)
    m = T.build_model({"model": {"arch": "temporal_unet", "image_size": [64, 64], "base_channels": 8}})
    x = torch.randn(1, 7, 64, 64)
    feats, _ = m.encode(x)
    out = m(x, feats.detach(), torch.zeros(1, 2, 64, 64), torch.ones(1, 1, 64, 64))
    out.abs().mean().backward()
    assert float(m.temporal_fuse.weight.grad.abs().sum()) > 0
    m.zero_grad()
    m(x).abs().mean().backward()
    g = m.temporal_fuse.weight.grad
    assert g is None or float(g.abs().sum()) == 0
