"""RCS metric behaviour — KB §11, Decision 53. Pure CPU, no session needed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.metrics.rcs import SequenceRCS, rcs, summarise

pytest.importorskip("torchvision")
from eval.metrics.rcs import AppearanceEncoder  # noqa: E402

H, W = 64, 64


@pytest.fixture(scope="module")
def enc():
    return AppearanceEncoder("cpu")


def _img(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((H, W, 3), dtype=np.float32)


def _mask(x0=8, x1=40) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[8:40, x0:x1] = True
    return m


def test_identical_appearance_scores_one(enc):
    img, m = _img(0), _mask()
    assert rcs(enc(img, m), enc(img, m)) == pytest.approx(1.0, abs=1e-5)


def test_different_appearance_scores_below_one(enc):
    m = _mask()
    assert rcs(enc(_img(1), m), enc(_img(2), m)) < 0.999


def test_only_the_masked_pixels_matter(enc):
    """Background must not reach the descriptor — otherwise RCS measures the
    scene changing behind an entity, not the entity."""
    img, m = _img(3), _mask()
    other = img.copy()
    other[~m] = _img(4)[~m]          # replace ALL background, keep the entity
    assert rcs(enc(img, m), enc(other, m)) == pytest.approx(1.0, abs=1e-5)


def test_changing_the_entity_does_move_the_score(enc):
    """Non-vacuity companion to the test above."""
    img, m = _img(5), _mask()
    other = img.copy()
    other[m] = 1.0 - other[m]
    assert rcs(enc(img, m), enc(other, m)) < 0.99


def test_embedding_is_unit_norm(enc):
    e = enc(_img(6), _mask())
    assert float(torch.linalg.norm(e)) == pytest.approx(1.0, abs=1e-5)


def test_empty_mask_raises_rather_than_returning_a_number(enc):
    with pytest.raises(ValueError):
        enc(_img(7), np.zeros((H, W), dtype=bool))


def test_score_is_bounded(enc):
    s = rcs(enc(_img(8), _mask()), enc(_img(9), _mask()))
    assert -1.0 <= s <= 1.0


def test_summarise_groups_by_duration_and_mechanism():
    rows = [
        SequenceRCS("a", "camera_away", 0.5, 1, 0, 1, 300, 300, 0.90),
        SequenceRCS("b", "camera_away", 8.0, 1, 0, 1, 300, 300, 0.80),
        SequenceRCS("c", "entity_away", 8.0, 1, 0, 1, 300, 300, 0.70),
    ]
    s = summarise(rows)
    assert s["overall"]["n"] == 3
    assert s["overall"]["mean"] == pytest.approx(0.80)
    assert s["by_unseen_seconds"]["8s"]["n"] == 2
    assert s["by_unseen_seconds"]["8s"]["mean"] == pytest.approx(0.75)
    assert s["by_mechanism"]["entity_away"]["mean"] == pytest.approx(0.70)
    # ordered by duration, so a reader sees the ladder in order
    assert list(s["by_unseen_seconds"]) == ["0.5s", "8s"]


def test_sd_is_none_for_a_single_sample():
    s = summarise([SequenceRCS("a", "camera_away", 1.0, 1, 0, 1, 300, 300, 0.9)])
    assert s["overall"]["sd"] is None, "a single sample has no sd; reporting 0.0 would mislead"
