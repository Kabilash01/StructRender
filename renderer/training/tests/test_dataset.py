"""Input-channel contract for ladder Model A (KB §4 CNN Stem, Decisions 8 and 9).

IN: depth(1) + normal(3) + albedo(3) = 7.  OUT: motion, entity_id.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data_pipeline.teacher.adapter import FrameStructure
from renderer.training.dataset import (
    CHANNEL_NAMES,
    IN_CHANNELS,
    StyledFrameDataset,
    build_input,
    inverse_depth,
)


def _synthetic(seed: int = 0) -> FrameStructure:
    rng = np.random.default_rng(seed)
    depth = rng.uniform(1.0, 20.0, (8, 12)).astype(np.float32)
    depth[0, :] = np.inf
    return FrameStructure(
        session_id="test",
        frame_id=0,
        depth=depth,
        normal=rng.uniform(-1, 1, (8, 12, 3)).astype(np.float32),
        albedo=rng.uniform(0, 1, (8, 12, 3)).astype(np.float32),
        entity_id=rng.integers(0, 5, (8, 12)).astype(np.int32),
    )


# --- depth normalization ---------------------------------------------------- #
def test_inverse_depth_maps_background_infinity_to_zero():
    d = np.array([[np.inf, 0.0, 1.0, 9.0]], dtype=np.float32)
    got = inverse_depth(d)
    assert got[0, 0] == 0.0
    assert np.allclose(got[0, 1:], [1.0, 0.5, 0.1])


def test_inverse_depth_is_bounded_and_monotone_in_nearness():
    d = np.linspace(0.0, 1000.0, 512, dtype=np.float32)
    got = inverse_depth(d)
    assert got.min() >= 0.0 and got.max() <= 1.0
    assert np.all(np.diff(got) <= 0.0)  # larger depth -> smaller value
    assert np.isfinite(got).all()


# --- channel composition ---------------------------------------------------- #
def test_input_has_exactly_seven_channels_in_the_declared_order():
    x = build_input(_synthetic())
    assert x.shape == (IN_CHANNELS, 8, 12) == (7, 8, 12)
    assert len(CHANNEL_NAMES) == 7
    fs = _synthetic()
    assert np.allclose(x[0], inverse_depth(fs.depth))
    assert np.allclose(x[1:4], np.transpose(fs.normal, (2, 0, 1)))
    assert np.allclose(x[4:7], np.transpose(fs.albedo, (2, 0, 1)))


def test_entity_id_is_not_an_input_channel(cfg=None):
    """KB Decision 9 — IDs are addresses. Model N is the control for raw-ID input;
    leaking IDs into Model A would destroy that control."""
    fs = _synthetic()
    base = build_input(fs)
    scrambled = FrameStructure(
        session_id=fs.session_id,
        frame_id=fs.frame_id,
        depth=fs.depth,
        normal=fs.normal,
        albedo=fs.albedo,
        entity_id=(fs.entity_id * 7919 + 13).astype(np.int32),
    )
    assert np.array_equal(base, build_input(scrambled))


def test_motion_is_not_an_input_channel():
    """KB §4 CNN Stem: 'motion is not treated as an unexplained appearance channel'."""
    import inspect

    from renderer.training import dataset as ds_mod

    src = inspect.getsource(ds_mod)
    assert '"motion"' not in src and "npz['motion']" not in src
    assert "motion" not in CHANNEL_NAMES


def test_input_is_finite_for_a_real_frame(session):
    x = build_input(session.load_structure(session.frame_ids()[0]))
    assert np.isfinite(x).all()
    assert x.shape[0] == 7


# --- dataset ---------------------------------------------------------------- #
def test_dataset_item_shapes_and_ranges(session, target_dir):
    ids = session.frame_ids()[:3]
    ds = StyledFrameDataset(session, target_dir, ids)
    x, y, fid = ds[0]
    h, w = session.resolution[1], session.resolution[0]
    assert x.shape == (7, h, w) and x.dtype == torch.float32
    assert y.shape == (3, h, w) and y.dtype == torch.float32
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0
    assert fid == ids[0]


def test_dataset_rejects_frames_without_a_target(session, target_dir):
    with pytest.raises(KeyError):
        StyledFrameDataset(session, target_dir, [10**9])
