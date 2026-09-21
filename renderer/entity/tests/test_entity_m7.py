"""M7 exit criteria, asserted rather than asserted-about.

  1. correct entity mask/token association
  2. no raw-ID production input
  3. overflow drops deterministically by the ranking rule, never arbitrary order
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from renderer.entity.keys import EntityKey, keys_for_frame
from renderer.entity.pooling import (EntityTokenProjection, entity_consistency_loss,
                                     masked_mean_pool)
from renderer.entity.selection import build_entity_masks, pixel_counts, select_entities


def _keys(*iids, gen=0):
    return {i: EntityKey("s", i, gen) for i in iids}


# ---- criterion 3: deterministic overflow ---------------------------------

def test_overflow_keeps_the_largest_entities():
    counts = {1: 10, 2: 500, 3: 50, 4: 900}
    assert select_entities(counts, _keys(1, 2, 3, 4), k_max=2) == [4, 2]


def test_overflow_is_by_pixel_count_not_insertion_order():
    counts = {9: 10, 1: 999}          # 9 inserted first, 1 is bigger
    assert select_entities(counts, _keys(9, 1), k_max=1) == [1]


def test_ties_break_deterministically_regardless_of_input_order():
    # equal pixel_count: the rule must still be total, or "deterministic" is a
    # claim about dict ordering rather than about the policy.
    a = select_entities({3: 100, 1: 100, 2: 100}, _keys(1, 2, 3), k_max=2)
    b = select_entities({2: 100, 3: 100, 1: 100}, _keys(3, 2, 1), k_max=2)
    assert a == b == [1, 2]


def test_generation_breaks_a_tie_after_instance_id():
    keys = {1: EntityKey("s", 1, 5), 2: EntityKey("s", 1, 2)}
    assert select_entities({1: 100, 2: 100}, keys, k_max=1) == [2]


def test_an_entity_without_a_key_is_dropped_not_given_a_slot():
    # no engine record => no address; a slot would hold an unidentified object
    assert select_entities({1: 999, 2: 5}, _keys(2), k_max=4) == [2]


def test_background_is_never_selected():
    m = np.zeros((4, 4), dtype=np.int32); m[0, 0] = 7
    assert 0 not in pixel_counts(m)


# ---- criterion 1: mask/token association ---------------------------------

def test_masks_land_in_selection_order_and_pad_with_zeros():
    idmap = np.array([[1, 1], [2, 0]], dtype=np.int32)
    masks = build_entity_masks(idmap, [2, 1], k_max=4)
    assert masks.shape == (4, 2, 2)
    assert masks[0].sum() == 1 and masks[1].sum() == 2      # slot order == selection order
    assert masks[2].sum() == 0 and masks[3].sum() == 0      # padding


def test_pooled_token_describes_only_its_own_entity():
    f = torch.zeros(1, 3, 2, 2)
    f[0, :, 0, :] = 1.0                                     # top row only
    m = torch.zeros(1, 2, 2, 2)
    m[0, 0, 0, :] = 1.0                                     # slot 0 = top row
    m[0, 1, 1, :] = 1.0                                     # slot 1 = bottom row
    p = masked_mean_pool(f, m)
    assert torch.allclose(p[0, 0], torch.ones(3))
    assert torch.allclose(p[0, 1], torch.zeros(3))


def test_changing_one_entitys_pixels_cannot_move_another_token():
    f = torch.randn(1, 8, 4, 4)
    m = torch.zeros(1, 2, 4, 4); m[0, 0, :2] = 1; m[0, 1, 2:] = 1
    before = masked_mean_pool(f, m).clone()
    f2 = f.clone(); f2[0, :, :2] += 9.0                     # disturb slot 0's pixels only
    after = masked_mean_pool(f2, m)
    assert not torch.allclose(before[0, 0], after[0, 0])
    assert torch.allclose(before[0, 1], after[0, 1])


def test_empty_slot_pools_to_zero_not_nan():
    p = masked_mean_pool(torch.randn(2, 5, 3, 3), torch.zeros(2, 3, 3, 3))
    assert torch.isfinite(p).all() and bool((p == 0).all())


def test_projection_starts_at_zero_so_entity_awareness_is_off_at_init():
    proj = EntityTokenProjection(6, 12)
    assert bool((proj(torch.randn(2, 4, 6)) == 0).all())


# ---- criterion 2: IDs are addresses, never inputs -------------------------

def test_entity_ids_never_enter_the_network_as_values():
    """Masks are binary indicators; the instance_id value must not survive."""
    small = build_entity_masks(np.array([[3]], dtype=np.int32), [3], k_max=1)
    large = build_entity_masks(np.array([[9999]], dtype=np.int32), [9999], k_max=1)
    assert np.array_equal(small, large)          # the ID's VALUE is gone
    assert set(np.unique(small)) <= {0.0, 1.0}


def test_key_parses_and_generation_separates_a_recycled_instance_id():
    k0 = EntityKey.parse("sess:7:0")
    k1 = EntityKey.parse("sess:7:1")
    assert (k0.instance_id, k0.generation) == (7, 0)
    assert k0.memory_slot != k1.memory_slot      # M8's no-leak rule rests on this


@pytest.mark.parametrize("bad", ["", "nope", "a:b", ":1:2"])
def test_malformed_keys_raise_rather_than_silently_mis_address(bad):
    with pytest.raises(ValueError):
        EntityKey.parse(bad)


def test_keys_for_frame_skips_destroyed_and_background():
    rows = [{"instance_id": 1, "entity_key": "s:1:0", "destroyed": False},
            {"instance_id": 2, "entity_key": "s:2:0", "destroyed": True},
            {"instance_id": 0, "entity_key": "s:0:0", "destroyed": False}]
    assert set(keys_for_frame(rows)) == {1}


# ---- consistency loss ----------------------------------------------------

def test_consistency_loss_ignores_slots_empty_in_either_frame():
    t, p = torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)
    p[0, 1] = 5.0                                  # slot 1 differs wildly
    m = torch.zeros(1, 2, 2, 2); m[0, 0] = 1       # but slot 1 is empty now
    pm = torch.ones(1, 2, 2, 2)
    assert float(entity_consistency_loss(t, m, p, pm)) == 0.0


def test_consistency_loss_penalises_a_drifting_occupied_slot():
    t, p = torch.zeros(1, 1, 4), torch.ones(1, 1, 4)
    m = torch.ones(1, 1, 2, 2)
    assert float(entity_consistency_loss(t, m, p, m)) == pytest.approx(1.0)


# ---- Model E wiring ------------------------------------------------------

def _kw():
    return dict(image_size=(64, 64), embed_dim=32, depths=(1, 1, 1),
                num_heads=(1, 1), window_sizes=(8, 4))


def test_model_e_is_identical_to_model_c_at_init():
    """Zero-init must make the ENTIRE entity branch vanish, not just the tokens.

    to_k/to_v carry biases, so zero tokens still give non-zero keys and values;
    only zeroing the attention's output projection makes the branch disappear.
    Without this, "did entity awareness help?" cannot be separated from "was it
    initialised differently?" — the confound Decision 66 found in Model B.
    """
    from renderer.entity.model import EntityAwareStructFormer
    from renderer.structformer.model import StructFormer
    torch.manual_seed(0)
    e = EntityAwareStructFormer(k_max=4, **_kw())
    c = StructFormer(**_kw())
    c.load_state_dict(e.backbone.state_dict())
    x = torch.randn(1, 7, 64, 64)
    st = torch.randn(1, c.style_dim)
    m = torch.zeros(1, 4, 64, 64)
    m[0, 0, :16, :16] = 1
    with torch.no_grad():
        assert torch.allclose(c(x, st), e(x, st), atol=1e-6)
        assert torch.allclose(c(x, st), e(x, st, m), atol=1e-6)


def test_model_e_entity_path_engages_once_trained():
    """The mirror of the test above: a branch that can never engage is dead code."""
    from renderer.entity.model import EntityAwareStructFormer
    torch.manual_seed(0)
    e = EntityAwareStructFormer(k_max=4, **_kw())
    x = torch.randn(1, 7, 64, 64)
    st = torch.randn(1, e.backbone.style_dim)
    m = torch.zeros(1, 4, 64, 64)
    m[0, 0, :16, :16] = 1
    with torch.no_grad():
        before = e(x, st, m)
        e.entity_attn.proj.weight.data.normal_(0, 0.1)
        assert not torch.allclose(before, e(x, st, m), atol=1e-6)


def test_one_entity_cannot_influence_another_at_the_attention_output():
    """M7 criterion 1, end to end: KB §4's -inf rule surviving the full model.

    The guarantee is asserted at the ATTENTION OUTPUT, not at the final image,
    and the distinction is real rather than a convenience. The decoder mixes
    spatially by design — convolutions and shifted windows deliberately spread
    information across the frame — so removing a distant entity DOES change a
    given pixel's final colour, through the decoder, long after attention is
    done. Asserting isolation on the output image would therefore fail for a
    correct model, and "fixing" it would mean crippling the decoder.

    What §4 actually promises is that an entity's TOKEN cannot reach a pixel
    outside its mask. That is exactly what is measured here.
    """
    from renderer.entity.model import EntityAwareStructFormer
    torch.manual_seed(0)
    e = EntityAwareStructFormer(k_max=4, **_kw())
    e.entity_attn.proj.weight.data.normal_(0, 0.1)      # make the branch live
    e.pool_proj.fc2.weight.data.normal_(0, 0.1)
    x = torch.randn(1, 7, 64, 64)
    st = torch.randn(1, e.backbone.style_dim)
    m = torch.zeros(1, 4, 64, 64)
    m[0, 0, :16, :16] = 1.0          # entity 0, top-left
    m[0, 1, 48:, 48:] = 1.0          # entity 1, bottom-right

    bb = e.backbone
    with torch.no_grad():
        h = bb.stem(x)
        g, b = bb.style_film(st).chunk(2, dim=-1)
        h = h * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        h, _ = bb.stage1(h)
        h, _ = bb.stage2(h)

        def attn(mask):
            mm = torch.nn.functional.interpolate(mask, size=h.shape[-2:], mode="area")
            mm = (mm > 0).to(h.dtype)
            return e.entity_attn(h, e._tokens(h, mm), mm), mm

        a1, mm = attn(m)
        m2 = m.clone()
        m2[0, 1, 48:, 48:] = 0.0     # delete entity 1 entirely
        a2, _ = attn(m2)

    own = mm[0, 0].bool()
    other = mm[0, 1].bool()
    assert int(own.sum()) > 0 and int(other.sum()) > 0, "degenerate masks"
    # entity 0 is untouched by entity 1 vanishing...
    assert torch.equal(a1[0, :, own], a2[0, :, own])
    # ...and the change is real where it should be, so this is not a no-op test
    assert not torch.allclose(a1[0, :, other], a2[0, :, other])


def test_single_style_form_takes_masks_but_not_a_style_input():
    """Decision 45's export contract: a constant style folds into a parameter;
    entity masks stay an input because they change every frame."""
    from renderer.entity.model import EntityAwareSingleStyle
    e = EntityAwareSingleStyle(k_max=4, **_kw())
    x = torch.randn(1, 7, 64, 64)
    m = torch.zeros(1, 4, 64, 64)
    with torch.no_grad():
        assert e(x, m).shape == (1, 3, 64, 64)
        assert e(x).shape == (1, 3, 64, 64)
