"""Model D's data unit — KB §10 Milestone 6, KB Decision 35's split.

The property that carries the milestone: a training sample may never reach across
the 60-frame guard band or out of the training split. Both splits' contiguity is
what makes a temporal rung trainable at all, and it is also what makes the
committed M6 gate genuinely held out (KB Decision 55) — so a sample that quietly
straddled the boundary would put gate frames into training.

Session-backed tests SKIP when the gitignored structural session is absent, the
same convention as the rest of the M6 suite.
"""

from __future__ import annotations

import pytest

from renderer.temporal.pair_dataset import (
    TemporalFrameDataset,
    TemporalTripleDataset,
    consecutive_triples,
)
from renderer.temporal.pairs import consecutive_pairs
from renderer.training.dataset import make_split, target_dir_for

VAL_SEQUENCES = ["reap_camera_away_8s", "reap_entity_away_8s"]
GUARD_FRAMES = 60


@pytest.fixture(scope="module")
def split(session):
    return make_split(session, VAL_SEQUENCES, GUARD_FRAMES)


@pytest.fixture(scope="module")
def targets(session):
    d = target_dir_for(session.session_id, "oil_impasto", None)
    if not d.is_dir():
        pytest.skip(f"no oil_impasto targets at {d}")
    return d


# --------------------------------------------------------------------------- #
# containment
# --------------------------------------------------------------------------- #
def test_no_triple_reaches_outside_the_training_split(session, split):
    """Every one of the three frames must be a training frame. The guard band and
    the validation sequences are both excluded from `train_ids`, so this single
    assertion covers both."""
    train = set(split.train_ids)
    triples = consecutive_triples(session, split.train_ids)
    assert triples
    outside = [t for t in triples
               if not {t.prev2_id, t.prev_id, t.curr_id} <= train]
    assert not outside, f"{len(outside)} triples reach outside the training split: {outside[:3]}"


def test_no_triple_straddles_a_guard_band_or_a_held_out_sequence(session, split):
    """Stated as its own test because it is the failure that would silently put
    gate frames into training: the guard band is where the pre-registered M6 gate
    lives (KB Decision 55), and its 28 pairs are held out of BOTH splits."""
    forbidden = set(split.val_ids) | set(split.dropped_guard_ids)
    touched = [t for t in consecutive_triples(session, split.train_ids)
               if {t.prev2_id, t.prev_id, t.curr_id} & forbidden]
    assert not touched, f"{len(touched)} triples touch a held-out frame: {touched[:3]}"


def test_triples_are_contiguous_and_one_tick_apart(session, split):
    """`dt` is engine TICKS, not frame ids — this session drops to a reduced rate
    during the long reappearance gaps, and an 8-tick step carries ~8x the
    displacement. Training is 100 % dt == 1 and the gate is a dt == 1 set."""
    for t in consecutive_triples(session, split.train_ids):
        assert t.prev_id == t.prev2_id + 1 and t.curr_id == t.prev_id + 1
        r0, r1, r2 = (session.record(i)["tick"] for i in (t.prev2_id, t.prev_id, t.curr_id))
        assert int(r1) - int(r0) == 1 and int(r2) - int(r1) == 1


def test_triples_are_built_by_chaining_the_existing_pair_builder(session, split):
    """Not a style preference: `consecutive_pairs` already carries the
    both-ends-inside and `motion_valid` guarantees, and re-deriving adjacency here
    would be a second implementation to keep in step with it."""
    pairs = {(p.prev_id, p.curr_id) for p in consecutive_pairs(session, split.train_ids)
             if p.tick_delta == 1}
    for t in consecutive_triples(session, split.train_ids):
        assert (t.prev2_id, t.prev_id) in pairs
        assert (t.prev_id, t.curr_id) in pairs


def test_a_frame_set_with_no_consecutive_runs_is_rejected_not_silently_empty(session, targets, split):
    """M4's overfit selection spreads frames evenly, which yields zero triples. A
    dataset that returned length 0 there would make the §11 pre-cloud gate pass
    vacuously."""
    scattered = split.train_ids[::10]
    with pytest.raises(ValueError, match="triples"):
        TemporalTripleDataset(session, targets, scattered)


# --------------------------------------------------------------------------- #
# item contract
# --------------------------------------------------------------------------- #
def test_a_triple_item_carries_seven_channel_structure_and_engine_side_masks(session, targets, split):
    ds = TemporalTripleDataset(session, targets, split.train_ids)
    item = ds[0]
    h, w = session.resolution[1], session.resolution[0]
    for k in ("structure_prev2", "structure_prev", "structure_curr"):
        assert tuple(item[k].shape) == (7, h, w), k
    assert tuple(item["motion_prev"].shape) == tuple(item["motion_curr"].shape) == (2, h, w)
    assert tuple(item["mask_prev"].shape) == tuple(item["mask_curr"].shape) == (1, h, w)
    assert tuple(item["target_curr"].shape) == (3, h, w)
    assert 0.0 <= float(item["mask_curr"].min()) and float(item["mask_curr"].max()) <= 1.0
    assert float(item["target_curr"].min()) >= 0.0 and float(item["target_curr"].max()) <= 1.0


def test_the_evaluation_dataset_scores_every_validation_frame(session, targets, split):
    """The §11 selection set must be the SAME 166 frames Models A and C were
    selected on (KB Decision 35). A triple-based validation set would have
    silently selected Model D on 40."""
    ds = TemporalFrameDataset(session, targets, split.val_ids)
    assert len(ds) == len(split.val_ids)
    assert ds.frame_ids == sorted(split.val_ids)
    assert len(consecutive_triples(session, split.val_ids)) < len(ds)


def test_a_frame_without_a_predecessor_is_served_as_a_cold_start(session, targets, split):
    """Zero mask, zero motion, own structure standing in — so the model's fusion
    gate makes the temporal contribution exactly zero without a branch."""
    ds = TemporalFrameDataset(session, targets, split.val_ids)
    cold = [i for i in range(len(ds)) if not ds[i]["has_history"]]
    assert cold, "expected at least the first frame of each held-out sequence"
    item = ds[cold[0]]
    assert float(item["mask_curr"].abs().sum()) == 0.0
    assert float(item["motion_curr"].abs().sum()) == 0.0
    assert item["structure_prev"].equal(item["structure_curr"])
