"""Model C's config against Model A's — KB §11 ladder integrity.

§11's ladder is only interpretable if consecutive rungs differ in ONE thing. M5's
brief is "Model C differs from Model A in architecture alone", and Decision 35
makes the split, the selection rule and the objective shared obligations across
every rung. Prose cannot enforce that; these tests compare the two configs
field-by-field, so a drift fails here rather than reaching a published comparison.

Pure JSON: no GPU, no session, no targets.
"""

from __future__ import annotations

import json

import pytest

from renderer.training.train_baseline import (
    LEGACY_REPORTS,
    build_model,
    load_config,
    objective_string,
    report_path,
)

from .conftest import ROOT

M4_CONFIG = ROOT / "configs" / "training" / "m4_unet_baseline.json"
M5_CONFIG = ROOT / "configs" / "structformer" / "m5_structformer.json"

#: The blocks that MUST be identical between the two rungs. `model` is excluded
#: because the architecture is the one intended difference; `train` is excluded
#: only for `checkpoint_dir` and is checked separately below.
SHARED_BLOCKS = ("input", "data", "split", "loss", "overfit", "style_acquisition")

#: KB §11 "Local Pre-Cloud Training Gate" item 4 — a literal, not a tunable.
OVERFIT_N_RANGE = (32, 128)


@pytest.fixture(scope="module")
def m4() -> dict:
    return load_config(M4_CONFIG)


@pytest.fixture(scope="module")
def m5() -> dict:
    return load_config(M5_CONFIG)


# --------------------------------------------------------------------------- #
# the rungs differ in architecture alone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block", SHARED_BLOCKS)
def test_shared_blocks_are_identical_to_model_a(block, m4, m5):
    """Byte-for-byte, not "equivalent". A reworded rationale is fine; a moved
    number is the confound Decision 35 exists to prevent."""
    assert json.dumps(m5[block], sort_keys=True) == json.dumps(m4[block], sort_keys=True)


def test_the_split_and_selection_rule_are_model_a_s(m4, m5):
    """Decision 35: contiguous sequence holdout, the same held-out sequences, the
    same 60-frame guard band. The selection rule (lowest val_l1) lives in the
    shared training loop, which is why both rungs run through `train_baseline`."""
    assert m5["split"]["policy"] == "contiguous_sequence_holdout"
    assert m5["split"]["val_sequences"] == m4["split"]["val_sequences"]
    assert m5["split"]["guard_frames"] == m4["split"]["guard_frames"] == 60


def test_the_objective_is_model_a_s_and_is_the_mandated_one(m4, m5):
    """Decision 41 mandates `1.0*L1 + 0.5*LPIPS(alex)` for EVERY §11 ladder model.
    Derived through the shared helper from the config's NUMBERS, never its prose
    (Decision 42) — the prose-splitting version of this once produced
    "L_rec = L1 + 0", which names the configuration Decision 41 records as
    FAILING the sharpness gate."""
    assert objective_string(m5) == objective_string(m4)
    assert objective_string(m5) == "1*L1 + 0.5*LPIPS(alex)"
    assert m5["loss"]["l1"] == 1.0
    assert m5["loss"]["perceptual"] == 0.5
    assert m5["loss"]["perceptual_net"] == "alex"


def test_the_training_schedule_is_model_a_s_apart_from_the_checkpoint_dir(m4, m5):
    """Seed included: if the seed differed it would be a second difference
    alongside the architecture, and Decision 46 measured that seed alone moves
    these metrics."""
    IGNORE = ("checkpoint_dir", "schedule_note", "batch_note", "batch_size", "accum_steps")
    a = {k: v for k, v in m4["train"].items() if k not in IGNORE}
    b = {k: v for k, v in m5["train"].items() if k not in IGNORE}
    assert b == a
    assert m5["train"]["checkpoint_dir"] != m4["train"]["checkpoint_dir"]

    # batch_size and accum_steps are exempted above ONLY because their product is
    # asserted equal here. Parameter parity puts Model C at embed 240, which OOMs
    # at batch 8, so it runs batch 4 x accum 2 (KB Decision 47). The optimiser
    # must still see the same number of samples per update as Model A, or batch
    # size becomes a second difference between the rungs — the precise thing this
    # test exists to prevent. Exempting them without this assertion would turn the
    # test into a rubber stamp.
    eff = lambda t: int(t["batch_size"]) * int(t.get("accum_steps", 1))
    assert eff(m5["train"]) == eff(m4["train"]) == 8, (
        f"effective batch differs: Model A {eff(m4['train'])}, Model C {eff(m5['train'])}"
    )


def test_model_c_is_the_frame_independent_rung(m5):
    """The ladder table is in §10 Milestone 9, NOT §11 (which has no lettered ladder).

    It places **C** = StructFormer baseline (Temporal No / Entity No / EPAM No).
    B is the CNN *temporal* rung — M6's deliverable — so calling this model B,
    as every M5 artifact first did, would have collided at the next milestone.
    KB Decision 50."""
    # Read from §10 Milestone 9's table rather than pinned as a literal: the
    # letter was wrong once (Decision 50) precisely because it was asserted
    # against itself. C is the row whose Backbone is StructFormer and whose
    # Temporal/Entity/EPAM columns are all No.
    from pathlib import Path
    kb = (Path(__file__).resolve().parents[3] / "docs" / "StructRender_KB.md").read_text(encoding="utf-8")
    row = next(l for l in kb.splitlines()
               if l.startswith("| C |") and "StructFormer" in l)
    assert "StructFormer baseline" in row, f"§10 M9 row C is not the StructFormer baseline: {row}"
    assert m5["ladder_model"] == "C", (
        "M5 delivers the frame-independent StructFormer, which §10 Milestone 9 calls C. "
        "B is the CNN *temporal* rung — M6's deliverable (KB Decision 50)."
    )
    assert m5["milestone"] == 5
    assert m5["model"]["arch"] == "structformer"


# --------------------------------------------------------------------------- #
# the mandatory local gate
# --------------------------------------------------------------------------- #
def test_the_overfit_gate_is_pre_registered_and_in_range(m4, m5):
    """§11's local pre-cloud gate is mandatory for this model too, and its
    thresholds are Model A's — imported unchanged so the gate cannot have been
    relaxed to let a new architecture through."""
    o5, o4 = m5["overfit"], m4["overfit"]
    lo, hi = OVERFIT_N_RANGE
    assert lo <= o5["n_frames"] <= hi
    assert o5["n_frames"] == o4["n_frames"] == 32
    assert o5["thresholds"] == o4["thresholds"]
    assert o5["thresholds"] == {
        "min_psnr_db": 30.0,
        "max_final_l1": 0.02,
        "min_loss_drop_factor": 10.0,
        "min_ssim": 0.95,
    }


# --------------------------------------------------------------------------- #
# artifact separation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(LEGACY_REPORTS))
def test_m5_reports_do_not_collide_with_m4(key, m4, m5):
    """An M5 run that overwrote an M4 artifact would destroy the reference half of
    the ladder comparison — the same class of defect as the overfit report that
    was overwritten on every test run (Decision 42)."""
    m5_path, m4_path = report_path(m5, key), report_path(m4, key)
    assert m5_path != m4_path
    assert "m4_" not in m5_path.name
    assert m5_path.parent == ROOT / "eval" / "reports"


def test_m5_declares_every_report_path(m5):
    """`report_path` falls back to M4's historical names for configs with no
    `reports` block. M5 must not rely on that fallback."""
    for key in LEGACY_REPORTS:
        assert key in m5["reports"], key


def test_m5_checkpoint_dir_is_not_m4_s(m5):
    assert "m4" not in m5["train"]["checkpoint_dir"]
    assert m5["train"]["checkpoint_dir"] == "checkpoints/m5_structformer"


# --------------------------------------------------------------------------- #
# the factory reads the config
# --------------------------------------------------------------------------- #
def test_the_factory_builds_each_rung_from_its_own_config(m4, m5):
    """One loop, two architectures. If `build_model` ignored `arch` both rungs
    would train the same network and the ladder would measure nothing."""
    from renderer.baselines.unet import UNetBaseline
    from renderer.structformer.model import StructFormerSingleStyle

    assert isinstance(build_model(m4), UNetBaseline)
    assert isinstance(build_model(m5), StructFormerSingleStyle)
