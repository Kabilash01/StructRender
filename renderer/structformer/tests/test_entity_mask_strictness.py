"""Entity masking must EXCLUDE, not merely discourage — KB Decision 52.

The module previously added a finite ``-100`` to illegal logits. That is a
penalty, not an exclusion: once the legal logits fall below the penalty, the
illegal keys dominate the softmax and one entity's memory reaches another
entity's pixels. These tests drive the real module (probe-only, never wired
into M5) on both attention paths and at logit magnitudes where a finite penalty
demonstrably fails.

Perturbations change ONE component of a token, or scale it. Adding a constant to
every component is removed by the module's ``norm_kv`` LayerNorm and would make
the test pass regardless of the masking.
"""

from __future__ import annotations

import pytest
import torch

from renderer.structformer.masked_entity_attention import EntityMaskedCrossAttention

DIM, K, H, W = 16, 4, 4, 6


def _module(impl: str, scale: float = 1.0) -> EntityMaskedCrossAttention:
    torch.manual_seed(0)
    m = EntityMaskedCrossAttention(
        dim=DIM, token_dim=DIM, num_heads=2, k_max=K, attn_impl=impl
    ).eval()
    # Blow up the logits: this is the regime where an additive -100 stops working.
    with torch.no_grad():
        m.to_q.weight.mul_(scale)
        m.to_k.weight.mul_(scale)
    return m


def _inputs():
    torch.manual_seed(1)
    x = torch.randn(1, DIM, H, W)
    tokens = torch.randn(1, K, DIM)
    mask = torch.zeros(1, K, H, W)
    mask[0, 0, :, :3] = 1.0          # entity 0 owns the left half
    mask[0, 1, :, 3:5] = 1.0         # entity 1 owns two columns
    # column 5 belongs to nobody; slot 2 and 3 are padded (never visible)
    return x, tokens, mask


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
@pytest.mark.parametrize("scale", [1.0, 50.0, 500.0])
def test_illegal_entity_cannot_influence_a_pixel(impl, scale):
    """Perturb entity 1's token; nothing in entity 0's region may move.

    At scale 500 the logits far exceed any finite penalty, which is exactly the
    case a -100 bias fails.
    """
    m = _module(impl, scale)
    x, tokens, mask = _inputs()
    with torch.no_grad():
        base = m(x, tokens, mask)
        moved = tokens.clone()
        moved[0, 1, 3] += 7.0        # ONE component: survives LayerNorm
        after = m(x, moved, mask)

    region0 = mask[0, 0].bool()
    delta = (after - base)[0][:, region0].abs().max().item()
    assert delta == 0.0, (
        f"entity 1's token moved entity 0's pixels by {delta:.3e} at scale {scale} "
        f"({impl}) — masking is a penalty, not an exclusion"
    )


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_padded_slots_cannot_influence_anything(impl):
    """Slots 2 and 3 are visible nowhere; scaling them must change no pixel."""
    m = _module(impl, 50.0)
    x, tokens, mask = _inputs()
    with torch.no_grad():
        base = m(x, tokens, mask)
        moved = tokens.clone()
        moved[0, 2] *= 9.0           # scaling survives LayerNorm's mean-centering
        moved[0, 3, 0] -= 11.0
        after = m(x, moved, mask)
    assert torch.equal(base, after), "a padded entity slot reached the output"


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_background_pixels_are_exactly_zero_and_finite(impl):
    """A pixel belonging to no entity must emit exactly 0.0 — not a blend of
    every entity's memory, and not NaN from an all -inf softmax row."""
    m = _module(impl, 50.0)
    x, tokens, mask = _inputs()
    with torch.no_grad():
        out = m(x, tokens, mask)
    background = out[0][:, :, 5]
    assert torch.isfinite(out).all(), "non-finite output — fully-masked row produced NaN"
    assert torch.equal(background, torch.zeros_like(background))


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_fully_masked_batch_stays_finite(impl):
    """Every row masked: the degenerate case that makes a naive -inf NaN."""
    m = _module(impl)
    x, tokens, _ = _inputs()
    empty = torch.zeros(1, K, H, W)
    with torch.no_grad():
        out = m(x, tokens, empty)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_legal_entity_does_still_influence_its_own_pixels():
    """Non-vacuity: the exclusion must not be excluding everything."""
    m = _module("manual", 1.0)
    x, tokens, mask = _inputs()
    with torch.no_grad():
        base = m(x, tokens, mask)
        moved = tokens.clone()
        moved[0, 0, 2] += 5.0
        after = m(x, moved, mask)
    region0 = mask[0, 0].bool()
    assert (after - base)[0][:, region0].abs().max() > 1e-6


def test_module_is_not_imported_by_the_M5_GRAPH():
    """M5's exported graph ships WITHOUT entity conditioning (KB Decisions 45/50).

    Model C must stay the clean frame-independent baseline the ladder compares
    against, so nothing in `renderer/structformer/` may pull entity machinery in.

    This **enumerates the package** rather than checking a hardcoded list of five
    filenames, which is how it was first written: a file added to the package
    later would have been silently unguarded, and the guard would have kept
    passing while the property it protects quietly stopped holding. It also
    parses each module and inspects real import statements instead of substring-
    matching the source, so a name appearing in a comment is not a false alarm
    and an aliased import is not a false pass.
    """
    import ast
    from pathlib import Path
    from renderer.training.train_baseline import ROOT

    pkg = Path(ROOT) / "renderer" / "structformer"
    banned = {"masked_entity_attention"}
    checked = []
    for path in sorted(pkg.glob("*.py")):
        if path.name in ("__init__.py", "masked_entity_attention.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            for n in names:
                assert not (set(n.split(".")) & banned), (
                    f"{path.name} imports entity machinery ({n}); "
                    "Model C must stay the clean baseline")
        checked.append(path.name)
    # the enumeration must actually have found the graph files, or this passes vacuously
    assert {"model.py", "encoder.py", "decoder.py"} <= set(checked), checked


def test_the_attention_module_is_still_reachable_and_still_the_reference():
    """Model E no longer imports this module — Decision 75 replaced the attention
    with `MaskedEntityInjection`, which is bit-identical on disjoint masks with
    the provably-dead Q/K removed.

    The module stays because it remains the REFERENCE the simplification is
    checked against: `test_injection_is_identical_to_masked_attention_on_disjoint_masks`
    asserts the two agree exactly, so if anyone ever changes the lookup, the
    equivalence claim fails loudly instead of quietly becoming false.
    """
    from renderer.entity.injection import MaskedEntityInjection  # noqa: F401
    from renderer.structformer.masked_entity_attention import (  # noqa: F401
        EntityMaskedCrossAttention)


def test_onnx_export_and_cpu_parity():
    """The masking change alters the exported graph (masked_fill/Where instead of
    an Add), so its ONNX behaviour is re-verified on CPU here.

    TensorRT behaviour after this change is NOT verified — that needs a GPU and a
    separate measurement (KB Decision 52). The historical probe artifact
    describes the OLD additive form and is preserved unchanged.
    """
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import io

    m = _module("manual", 50.0)
    x, tokens, mask = _inputs()
    with torch.no_grad():
        ref = m(x, tokens, mask).numpy()

    buf = io.BytesIO()
    torch.onnx.export(
        m, (x, tokens, mask), buf,
        input_names=["x", "entity_tokens", "entity_mask"], output_names=["out"],
        opset_version=17, dynamo=False, do_constant_folding=True,
    )
    graph = onnx.load_from_string(buf.getvalue()).graph
    dims = [d for vi in list(graph.input) + list(graph.output)
            for d in vi.type.tensor_type.shape.dim]
    assert all(d.HasField("dim_value") and d.dim_value > 0 for d in dims), "symbolic dims"

    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": x.numpy(), "entity_tokens": tokens.numpy(),
                          "entity_mask": mask.numpy()})[0]
    assert torch.isfinite(torch.from_numpy(got)).all(), "ONNX output non-finite"
    assert abs(got - ref).max() <= 1e-3, f"CPU ONNX parity {abs(got - ref).max():.3e}"
