"""M4 exit criteria 1 and 3 against the actual trained checkpoint.

Skips (never fails) when ``checkpoints/m4_unet_baseline/best.pt`` is absent — the
checkpoint directory is gitignored, so a clean checkout has nothing to check.

Criterion 3, "recognizable styled output", is made falsifiable rather than left
to eyeballing. Three things must hold, because the first two alone are satisfied
by a blur:

1. the prediction is closer to the teacher target than the engine's own unstyled
   render is;
2. it travels at least ``min_hist_travel_fraction_of_target`` of the target's
   colour-histogram distance away from the native palette — the quantity M3's
   gate 4 uses to reject a passthrough (KB Decision 32, Finding 1);
3. it keeps at least ``min_sharpness_ratio`` of the target's high-frequency
   energy. (2) alone rewards smoothing: a blurred frame keeps its shifted
   palette, so its histogram travel goes *up*.

Thresholds live in the config's ``style_acquisition`` block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from renderer.training.dataset import StyledFrameDataset, make_split
from renderer.training.train_baseline import ROOT, evaluate, load_checkpoint, style_acquisition


@pytest.fixture(scope="module")
def trained(cfg, session, target_dir, device):
    ckpt = Path(ROOT) / cfg["train"]["checkpoint_dir"] / "best.pt"
    if not ckpt.is_file():
        pytest.skip(f"no trained checkpoint at {ckpt} — run train_baseline --mode train")
    model = load_checkpoint(ckpt, device)
    split = make_split(session, cfg["split"]["val_sequences"], int(cfg["split"]["guard_frames"]))
    val = StyledFrameDataset(session, target_dir, split.val_ids)
    return model, val, split, device


def test_training_report_shows_a_decreasing_loss(cfg):
    """Criterion 1: 'model trains'."""
    report = Path(ROOT) / "eval" / "reports" / "m4_train.json"
    if not report.is_file():
        pytest.skip("no training report yet")
    hist = json.loads(report.read_text(encoding="utf-8"))["history"]
    assert hist[-1]["train_l1"] < 0.5 * hist[0]["train_l1"]
    vals = [h["val_l1"] for h in hist if "val_l1" in h]
    assert len(vals) >= 2 and min(vals) < vals[0]


def test_held_out_metrics_are_recorded_with_the_split(cfg):
    report = Path(ROOT) / "eval" / "reports" / "m4_metrics.json"
    if not report.is_file():
        pytest.skip("no metrics report yet")
    r = json.loads(report.read_text(encoding="utf-8"))
    assert r["split"]["policy"] == "contiguous_sequence_holdout"
    assert r["split"]["val_sequences"] == cfg["split"]["val_sequences"]
    assert r["split"]["guard_frames"] == cfg["split"]["guard_frames"]
    assert {"l1", "psnr_db", "ssim"} <= set(r["selection_set"])


def test_reported_metrics_are_labelled_as_the_selection_set(cfg):
    """The val split is also the model-selection set (``best.pt`` is chosen on its
    lowest ``val_l1``), so these numbers are the selection criterion's own
    minimum. Calling them ``held_out`` overstates them; the report must say so."""
    report = Path(ROOT) / "eval" / "reports" / "m4_metrics.json"
    if not report.is_file():
        pytest.skip("no metrics report yet")
    r = json.loads(report.read_text(encoding="utf-8"))
    assert "held_out" not in r, "'held_out' overstates a set that is also the selection set"
    assert "selection_set" in r and "selection_set_note" in r
    assert "selection" in r["selection_set_note"].lower()


@pytest.fixture(scope="module")
def style(trained):
    model, val, _, device = trained
    return style_acquisition(model, val, device)


def test_output_beats_the_do_nothing_anchor(style):
    """Criterion 3(1)."""
    assert style["psnr_pred_vs_target_db"] > style["psnr_native_vs_target_db"], (
        f"prediction ({style['psnr_pred_vs_target_db']:.2f} dB) is no closer to the styled target "
        f"than the unstyled engine frame ({style['psnr_native_vs_target_db']:.2f} dB)"
    )


def test_output_is_styled_not_a_native_passthrough(cfg, style):
    """Criterion 3(2). ``min_hist_travel_fraction_of_target`` is what makes the
    prose word "comparable" mean something; at the old 0.5 the assertion admitted
    a prediction that travelled half as far as the target."""
    frac = cfg["style_acquisition"]["min_hist_travel_fraction_of_target"]
    assert style["hist_pred_vs_native"] >= frac * style["hist_target_vs_native"], (
        f"prediction travelled only {style['hist_pred_vs_native']:.3f} from the native palette "
        f"vs the target's {style['hist_target_vs_native']:.3f} (need {frac}x) — looks like a passthrough"
    )


def test_output_is_not_blurrier_than_the_target(cfg, style):
    """Criterion 3(3) — the check the histogram term cannot do.

    Histogram travel *rises* under smoothing, so ``hist_pred > hist_target`` is
    equally consistent with "blurrier than the target". High-frequency energy is
    not: blur drives it down.
    """
    floor = cfg["style_acquisition"]["min_sharpness_ratio"]
    assert style["sharpness_ratio"] >= floor, (
        f"prediction carries only {style['sharpness_ratio']:.3f} of the target's high-frequency "
        f"energy (need {floor}) — it is reaching the target by blurring"
    )


def test_held_out_output_is_not_degenerate(trained):
    """A constant image would already score well on L1 in this low-contrast scene,
    so require the prediction to (a) carry spatial contrast and (b) change between
    frames *as much as the targets do*.

    The frame-to-frame check is relative, not absolute: some held-out frames are
    genuinely near-identical (the reappearance rungs hold the camera still), so a
    fixed threshold would be measuring the scene, not the model.
    """
    model, val, _, device = trained
    m = evaluate(model, val, device)
    assert m["ssim"] > 0.3

    idxs = [0, len(val) // 4, len(val) // 2, 3 * len(val) // 4, len(val) - 1]
    with torch.no_grad():
        preds = torch.stack([model(val[i][0][None].to(device))[0].float().cpu() for i in idxs])
    targs = torch.stack([val[i][1] for i in idxs])
    assert float(preds.std()) > 0.02, "prediction is a flat image"

    pairs = [(a, b) for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
    d_pred = sum(float((preds[a] - preds[b]).abs().mean()) for a, b in pairs) / len(pairs)
    d_targ = sum(float((targs[a] - targs[b]).abs().mean()) for a, b in pairs) / len(pairs)
    assert d_targ > 0.005, "held-out frames are near-duplicates — the split is not measuring anything"
    assert d_pred >= 0.5 * d_targ, (
        f"predictions vary {d_pred:.4f} between frames while the targets vary {d_targ:.4f} "
        "— the model is close to emitting one constant image"
    )
