"""The train/val split is the trap of M4: all 990 frames come from ONE session at
60 fps, so a random frame split leaks near-duplicate neighbours into train and
makes val metrics meaningless (same class of defect as KB Decision 33).

These tests assert the split is by contiguous sequence AND that train/val are
non-adjacent, not merely disjoint.
"""

from __future__ import annotations

import pytest

from renderer.training.dataset import make_split


@pytest.fixture(scope="module")
def split(cfg, session):
    return make_split(session, cfg["split"]["val_sequences"], int(cfg["split"]["guard_frames"]))


def test_val_sequences_are_whole_contiguous_sequences(cfg, session, split):
    by_id = {s["sequence_id"]: s for s in session.header["sequences"]}
    expected = set()
    for name in cfg["split"]["val_sequences"]:
        lo, hi = by_id[name]["frame_range"]
        expected |= set(range(lo, hi + 1))
    assert set(split.val_ids) == expected & set(session.frame_ids())


def test_train_and_val_are_disjoint(split):
    assert set(split.train_ids).isdisjoint(split.val_ids)


def test_train_and_val_are_non_adjacent(cfg, split):
    """No train frame may sit within ``guard_frames`` of any val frame."""
    guard = int(cfg["split"]["guard_frames"])
    assert guard > 0
    val = sorted(split.val_ids)
    worst = min(min(abs(t - v) for v in val) for t in split.train_ids)
    assert worst > guard, f"closest train frame is {worst} frames from a val frame (guard={guard})"


def test_split_covers_the_session_without_overlap(session, split):
    all_ids = set(session.frame_ids())
    parts = [set(split.train_ids), set(split.val_ids), set(split.dropped_guard_ids)]
    assert set().union(*parts) == all_ids
    assert sum(len(p) for p in parts) == len(all_ids)


def test_split_is_deterministic(cfg, session):
    a = make_split(session, cfg["split"]["val_sequences"], int(cfg["split"]["guard_frames"]))
    b = make_split(session, cfg["split"]["val_sequences"], int(cfg["split"]["guard_frames"]))
    assert a.train_ids == b.train_ids and a.val_ids == b.val_ids


def test_both_splits_are_non_trivial(split):
    assert len(split.val_ids) >= 100
    assert len(split.train_ids) >= 400


def test_unknown_sequence_is_an_error(session):
    with pytest.raises(KeyError):
        make_split(session, ["no_such_sequence"], 60)
