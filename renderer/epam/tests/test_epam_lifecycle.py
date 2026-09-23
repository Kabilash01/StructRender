"""M8 exit criteria 1-3 and every §4 EPAM boundary, asserted."""

from __future__ import annotations

import pytest
import torch

from renderer.entity.keys import EntityKey
from renderer.epam.memory import EPAM, visibility_confidence

S = "sess"


def k(i, gen=0, s=S):
    return EntityKey(s, i, gen)


def f(v, d=4):
    return torch.full((d,), float(v))


# ---- criterion 1: same entity retrieves same memory after leaving view ----

def test_memory_is_frozen_while_out_of_view_and_returned_on_reappearance():
    m = EPAM(S, "oil")
    m.observe(k(1), f(1.0), 0.9, frame=10)
    before = m.lookup(k(1)).embedding.clone()
    # frames 11..500: entity not visible — nothing is called for it at all
    m.observe(k(2), f(5.0), 0.9, frame=200)          # other entities keep living
    e = m.lookup(k(1))
    assert torch.equal(e.embedding, before)
    assert e.last_seen_frame == 10


def test_reappearance_updates_the_same_entry_not_a_new_one():
    m = EPAM(S, "oil", gamma=0.5)
    m.observe(k(1), f(0.0), 0.9, frame=0)
    m.observe(k(1), f(2.0), 0.9, frame=300)
    e = m.lookup(k(1))
    assert torch.allclose(e.embedding, f(1.0))       # 0.5*0 + 0.5*2
    assert (e.first_seen_frame, e.last_seen_frame) == (0, 300)
    assert len(m) == 1


# ---- criterion 2: destroyed entity cannot leak memory to a later object ----

def test_destroy_deletes():
    m = EPAM(S, "oil")
    m.observe(k(1), f(1.0), 0.9, frame=0)
    m.destroy(k(1))
    assert m.lookup(k(1)) is None and len(m) == 0


def test_recycled_instance_id_does_not_inherit_the_destroyed_entitys_memory():
    m = EPAM(S, "oil")
    m.observe(k(7, gen=0), f(1.0), 0.9, frame=0)
    m.destroy(k(7, gen=0))
    assert m.lookup(k(7, gen=1)) is None
    m.observe(k(7, gen=1), f(9.0), 0.9, frame=50)
    assert torch.equal(m.lookup(k(7, gen=1)).embedding, f(9.0))   # fresh, not blended


def test_recycled_id_is_separate_even_if_destroy_event_was_missed():
    """generation, not the destroy call, is what guarantees no leak."""
    m = EPAM(S, "oil")
    m.observe(k(7, gen=0), f(1.0), 0.9, frame=0)
    assert m.lookup(k(7, gen=1)) is None


def test_destroy_of_unknown_entity_is_not_an_error():
    EPAM(S, "oil").destroy(k(3))


# ---- criterion 3: style change clears memory ----

def test_style_change_clears():
    m = EPAM(S, "oil")
    m.observe(k(1), f(1.0), 0.9, frame=0)
    m.set_style("watercolor")
    assert len(m) == 0 and m.lookup(k(1)) is None


def test_setting_the_same_style_does_not_clear():
    m = EPAM(S, "oil")
    m.observe(k(1), f(1.0), 0.9, frame=0)
    m.set_style("oil")
    assert m.lookup(k(1)) is not None


def test_an_entry_from_another_style_is_never_returned():
    """Defence in depth: even if a clear were skipped, lookup refuses it."""
    m = EPAM(S, "oil")
    m.observe(k(1), f(1.0), 0.9, frame=0)
    m.style_id = "watercolor"                       # simulate a missed clear
    assert m.lookup(k(1)) is None


# ---- update rule and confidence gating ----

def test_low_confidence_observation_touches_nothing():
    m = EPAM(S, "oil", tau=0.5)
    m.observe(k(1), f(1.0), 0.9, frame=0)
    m.observe(k(1), f(99.0), 0.2, frame=5)
    e = m.lookup(k(1))
    assert torch.equal(e.embedding, f(1.0)) and e.last_seen_frame == 0


def test_no_entry_is_created_from_an_unreliable_first_sighting():
    m = EPAM(S, "oil", tau=0.5)
    m.observe(k(1), f(1.0), 0.3, frame=0)
    assert m.lookup(k(1)) is None


def test_ema_converges_toward_a_repeated_observation():
    m = EPAM(S, "oil", gamma=0.9)
    m.observe(k(1), f(0.0), 0.9, frame=0)
    for t in range(1, 200):
        m.observe(k(1), f(1.0), 0.9, frame=t)
    assert torch.allclose(m.lookup(k(1)).embedding, f(1.0), atol=1e-6)


def test_memory_is_detached_from_the_graph():
    m = EPAM(S, "oil")
    x = torch.ones(4, requires_grad=True)
    m.observe(k(1), x * 2, 0.9, frame=0)
    assert not m.lookup(k(1)).embedding.requires_grad


# ---- session boundaries ----

def test_a_key_from_another_session_is_rejected():
    with pytest.raises(ValueError):
        EPAM(S, "oil").observe(k(1, s="other"), f(1.0), 0.9, frame=0)


def test_end_session_clears():
    m = EPAM(S, "oil")
    m.observe(k(1), f(1.0), 0.9, frame=0)
    m.end_session()
    assert len(m) == 0


def test_no_eviction_every_live_entity_is_kept():
    m = EPAM(S, "oil")
    for i in range(1, 501):
        m.observe(k(i), f(i), 0.9, frame=i)
    assert len(m) == 500


def test_visibility_confidence_is_bounded():
    assert visibility_confidence(0) == 0.0
    assert visibility_confidence(10**6) == 1.0
