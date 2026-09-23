"""Model F wiring and the shared memory step."""

from __future__ import annotations

import torch

from renderer.epam.model import StructRenderF, key_code
from renderer.epam.runtime import memory_step
from renderer.epam.train_epam import clip_forward

BIG = 5000     # px: reliable (conf 1.0)
SMALL = 100    # px: unreliable (conf 0.06)


def _kw():
    return dict(image_size=(64, 64), embed_dim=32, depths=(1, 1, 1),
                num_heads=(1, 1), window_sizes=(8, 4))


def test_memory_step_create_then_ema():
    st = [dict()]
    keys = torch.tensor([[key_code(1, 0)]])
    memory_step(st, keys, torch.zeros(1, 1, 2), torch.tensor([[BIG]]), 0.5, 0.5)
    out = memory_step(st, keys, torch.full((1, 1, 2), 2.0), torch.tensor([[BIG]]), 0.5, 0.5)
    assert torch.allclose(out, torch.full((1, 1, 2), 1.0))


def test_memory_step_follows_the_entity_across_a_slot_swap():
    """Decision 76 at the runtime level: memory is keyed, not slotted."""
    a, b = key_code(1, 0), key_code(2, 0)
    st = [dict()]
    memory_step(st, torch.tensor([[a, b]]), torch.tensor([[[0.0], [10.0]]]),
                torch.tensor([[BIG, BIG]]), 0.5, 0.5)
    out = memory_step(st, torch.tensor([[b, a]]), torch.tensor([[[10.0], [0.0]]]),
                      torch.tensor([[BIG, BIG]]), 0.5, 0.5)
    assert torch.equal(out, torch.tensor([[[10.0], [0.0]]]))   # each blended with ITS OWN past


def test_memory_step_unreliable_observation_uses_memory_and_stores_nothing():
    k1 = torch.tensor([[key_code(1, 0)]])
    st = [dict()]
    memory_step(st, k1, torch.ones(1, 1, 2), torch.tensor([[BIG]]), 0.9, 0.5)
    out = memory_step(st, k1, torch.full((1, 1, 2), 9.0), torch.tensor([[SMALL]]), 0.9, 0.5)
    assert torch.equal(out, torch.ones(1, 1, 2))
    assert torch.equal(st[0][int(k1)], torch.ones(2))


def test_absent_entity_memory_is_kept_unchanged():
    a, b = key_code(1, 0), key_code(2, 0)
    st = [dict()]
    memory_step(st, torch.tensor([[a]]), torch.ones(1, 1, 2), torch.tensor([[BIG]]), 0.9, 0.5)
    memory_step(st, torch.tensor([[b]]), torch.zeros(1, 1, 2), torch.tensor([[BIG]]), 0.9, 0.5)
    assert torch.equal(st[0][a], torch.ones(2))


def test_store_is_detached_but_output_keeps_grad():
    cur = torch.ones(1, 1, 2, requires_grad=True)
    st = [dict()]
    out = memory_step(st, torch.tensor([[key_code(1, 0)]]), cur, torch.tensor([[BIG]]), 0.9, 0.5)
    assert out.requires_grad and not next(iter(st[0].values())).requires_grad


def _clip(T=3):
    torch.manual_seed(0)
    m = torch.zeros(1, T, 4, 64, 64); m[:, :, 0, :32, :32] = 1
    keys = torch.zeros(1, T, 4, dtype=torch.long); keys[:, :, 0] = key_code(1, 0)
    return {"structure": torch.randn(1, T, 7, 64, 64), "masks": m, "keys": keys,
            "px": torch.full((1, T, 4), BIG), "style": torch.tensor([1])}


def test_gamma_zero_is_exactly_the_single_frame_model():
    """The ablation rests on this: F0 renders the last frame from that frame alone."""
    torch.manual_seed(0)
    f = StructRenderF(["a", "b"], k_max=4, **_kw()).eval()
    f.entity_attn.proj.weight.data.normal_(0, 0.1); f.pool_proj.fc2.weight.data.normal_(0, 0.1)
    b = _clip()
    with torch.no_grad():
        clip = clip_forward(f, b, 0.0, 0.5, torch.device("cpu"), grad_last=False)
        single = f(b["structure"][:, -1], b["style"], b["masks"][:, -1])
    assert torch.equal(clip, single)


def test_gamma_nonzero_actually_uses_history():
    """The mirror: if memory changed nothing, F would be F0 under another name."""
    torch.manual_seed(0)
    f = StructRenderF(["a", "b"], k_max=4, **_kw()).eval()
    f.entity_attn.proj.weight.data.normal_(0, 0.1); f.pool_proj.fc2.weight.data.normal_(0, 0.1)
    b = _clip()
    with torch.no_grad():
        f0 = clip_forward(f, b, 0.0, 0.5, torch.device("cpu"), grad_last=False)
        f9 = clip_forward(f, b, 0.9, 0.5, torch.device("cpu"), grad_last=False)
    assert not torch.allclose(f0, f9)


def test_styles_render_differently():
    torch.manual_seed(0)
    f = StructRenderF(["a", "b"], k_max=4, **_kw()).eval()
    x = torch.randn(1, 7, 64, 64); m = torch.zeros(1, 4, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(f(x, torch.tensor([0]), m), f(x, torch.tensor([1]), m))
