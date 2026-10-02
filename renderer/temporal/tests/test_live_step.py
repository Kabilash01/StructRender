"""LiveStep — the exported live temporal step of two-style B (KB Decision 97)."""

import numpy as np
import torch

from renderer.temporal.confidence import confidence_mask
from renderer.temporal.live_step import LiveStep, geometry_from_structure
from renderer.temporal.style_unet import StyleTemporalUNet
from renderer.training.dataset import inverse_depth


def _model():
    torch.manual_seed(0)
    m = StyleTemporalUNet(["oil", "wc"], image_size=(64, 64), base_channels=8).eval()
    for f in m.film:
        torch.nn.init.normal_(f.weight, 0, 0.1)
    torch.nn.init.normal_(m.temporal_fuse.weight, 0, 0.05)
    return m


def _x(seed):
    g = torch.Generator().manual_seed(seed)
    depth = 1 + 20 * torch.rand(64, 64, generator=g)
    depth[:8] = float("inf")                                  # sky rows
    x = torch.rand(1, 7, 64, 64, generator=g)
    x[0, 0] = torch.from_numpy(inverse_depth(depth.numpy()))
    x[0, 1:4, :8] = 0                                         # background normals are 0
    return x, depth


def test_geometry_round_trip_matches_engine_depth():
    x, depth = _x(1)
    d, finite, n = geometry_from_structure(x)
    fin = torch.isfinite(depth)
    assert torch.equal(finite[0, 0].bool(), fin)
    assert float((d[0, 0][fin] - depth[fin]).abs().max()) < 1e-4    # metres, vs a 0.175 m mask scale
    assert float(d[0, 0][~fin].abs().max()) == 0.0


@torch.no_grad()
def test_no_history_is_exactly_the_cold_start_model():
    m = _model()
    x, _ = _x(2)
    xp, _ = _x(3)
    for s in (0, 1):
        rgb, feat = LiveStep(m, s)(x, xp, torch.randn(1, *m.encode(x)[0].shape[1:]), torch.randn(1, 2, 64, 64) * 0.01,
                                   torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 1, 1))
        assert torch.equal(rgb, m(x, style_idx=torch.tensor([s])))
        assert torch.equal(feat, m.encode(x)[0])


@torch.no_grad()
def test_with_history_equals_the_scored_forward_with_the_training_mask():
    m = _model()
    x, depth = _x(4)
    xp, depth_p = _x(5)
    motion = torch.randn(1, 2, 64, 64) * 0.01
    no_prev = torch.zeros(1, 1, 64, 64); no_prev[..., 30:34, 30:34] = 1
    fin, finp = torch.isfinite(depth), torch.isfinite(depth_p)
    mask = confidence_mask(
        depth_t=torch.where(fin, depth, 0.)[None, None], depth_prev=torch.where(finp, depth_p, 0.)[None, None],
        normal_t=x[:, 1:4], normal_prev=xp[:, 1:4], motion=motion,
        depth_finite_t=fin.float()[None, None], depth_finite_prev=finp.float()[None, None],
        motion_valid=True, no_prev_transform=no_prev)["mask"]
    fp, _ = m.encode(xp)
    rgb, _ = LiveStep(m, 1)(x, xp, fp, motion, no_prev, torch.ones(1, 1, 1, 1))
    ref = m(x, fp, motion, mask, style_idx=torch.tensor([1]))
    assert float((rgb - ref).abs().max()) < 1e-5
