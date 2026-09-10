"""Model D's config against Model C's — KB §11 ladder integrity.

§11's ladder is only interpretable if consecutive rungs differ in ONE thing. M6's
intended difference is the temporal reprojection and the loss term that supervises
it; Decision 35 makes the split, the selection rule and the objective shared
obligations across every rung. Prose cannot enforce that, so these tests compare
the two configs field by field — the same construction
`renderer/structformer/tests/test_m5_config.py` uses for C against A.

Pure JSON: no GPU, no session, no targets.
"""

from __future__ import annotations

import json

import pytest

from renderer.training.train_baseline import (
    LEGACY_REPORTS,
    SUPPORTED_REPORT_KEYS,
    load_config,
    objective_string,
    report_path,
)

from .conftest import ROOT

M4_CONFIG = ROOT / "configs" / "training" / "m4_unet_baseline.json"
M5_CONFIG = ROOT / "configs" / "structformer" / "m5_structformer.json"
M6_CONFIG = ROOT / "configs" / "temporal" / "m6_temporal.json"

#: Blocks that must be byte-identical to Model C's. `loss` is excluded because
#: the temporal term is M6's deliverable and is checked separately; `model` and
#: `train` are excluded and checked field by field below.
SHARED_BLOCKS = ("input", "data", "split", "style_acquisition")

#: The only `train` keys allowed to differ from Model C's, and the only ones
#: allowed to be new. Anything else appearing here is a second difference between
#: two rungs the ladder intends to differ in architecture alone.
TRAIN_CHANGED = {"checkpoint_dir", "batch_note"}
TRAIN_ADDED = {"init_from", "init_from_note"}
TRAIN_REMOVED = {"schedule_note"}


@pytest.fixture(scope="module")
def m4() -> dict:
    return load_config(M4_CONFIG)


@pytest.fixture(scope="module")
def m5() -> dict:
    return load_config(M5_CONFIG)


@pytest.fixture(scope="module")
def m6() -> dict:
    return load_config(M6_CONFIG)


def test_it_declares_itself_as_the_temporal_rung(m6):
    assert m6["config_version"] == "m6-temporal-1"
    assert m6["milestone"] == 6
    assert m6["ladder_model"] == "D"
    assert m6["model"]["arch"] == "temporal_structformer"


# --------------------------------------------------------------------------- #
# the rungs differ in the temporal mechanism alone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block", SHARED_BLOCKS)
def test_shared_blocks_are_identical_to_model_c(block, m5, m6):
    """Byte-for-byte, not "equivalent". A reworded rationale is fine; a moved
    number is the confound Decision 35 exists to prevent."""
    assert json.dumps(m6[block], sort_keys=True) == json.dumps(m5[block], sort_keys=True)


def test_the_architecture_block_is_model_c_s_plus_temporal_prose(m5, m6):
    """Every architecture NUMBER must be Model C's — width, depth, heads, windows,
    attention and shift implementations. Model D adds no architecture knobs; if
    one appeared, the ladder would be measuring two changes at once."""
    for k, v in m5["model"].items():
        if k == "arch":
            continue
        assert m6["model"][k] == v, k
    added = set(m6["model"]) - set(m5["model"])
    assert added == {"temporal_fusion", "temporal_fusion_note",
                     "parameter_parity_note", "prev_feature_note"}, added


def test_the_split_and_selection_rule_are_model_c_s(m4, m5, m6):
    """Decision 35: contiguous sequence holdout, the same held-out sequences, the
    same 60-frame guard band. The selection rule (lowest val_l1) lives in the
    training loop; `train_temporal.evaluate` returns `l1` on the SAME 166
    validation frames, so Model D is selected on the same set as A and C."""
    assert m6["split"]["policy"] == "contiguous_sequence_holdout"
    assert m6["split"]["val_sequences"] == m5["split"]["val_sequences"] == m4["split"]["val_sequences"]
    assert m6["split"]["guard_frames"] == m5["split"]["guard_frames"] == 60


def test_the_reconstruction_objective_is_the_mandated_one_untouched(m4, m5, m6):
    """Decision 41 mandates `1.0*L1 + 0.5*LPIPS(alex)` for EVERY §11 ladder model.
    The temporal term is an ADDITION; the reconstruction weights may not move to
    make room for it."""
    assert objective_string(m6) == objective_string(m5) == objective_string(m4)
    assert objective_string(m6) == "1*L1 + 0.5*LPIPS(alex)"
    assert m6["loss"]["l1"] == m5["loss"]["l1"] == 1.0
    assert m6["loss"]["perceptual"] == m5["loss"]["perceptual"] == 0.5
    assert m6["loss"]["perceptual_net"] == m5["loss"]["perceptual_net"] == "alex"
    assert m6["loss"]["notes"] == m5["loss"]["notes"]


def test_the_loss_block_adds_the_temporal_term_and_nothing_else(m5, m6):
    added = set(m6["loss"]) - set(m5["loss"])
    assert added == {"temporal", "temporal_note", "temporal_weight_note"}, added
    assert m6["loss"]["temporal"] > 0.0


def test_the_temporal_weight_is_declared_untuned(m6):
    """Deliverable, stated as a test: the weight was NOT tuned against the M6
    gate. A hyper-parameter swept against a pre-registered exit criterion stops
    being pre-registered."""
    note = m6["loss"]["temporal_weight_note"].upper()
    assert "NOT TUNED" in note


def test_the_training_schedule_is_model_c_s_apart_from_the_declared_differences(m5, m6):
    """Seed, epochs, LR, weight decay, AMP dtype and eval cadence are Model C's.
    `batch_size`/`accum_steps` are exempt ONLY because their product is asserted
    equal — the exemption without the assertion would be a rubber stamp (KB
    Decision 47)."""
    a, b = m5["train"], m6["train"]
    assert set(a) - set(b) == TRAIN_REMOVED
    assert set(b) - set(a) == TRAIN_ADDED
    differ = {k for k in set(a) & set(b) if a[k] != b[k]}
    assert differ <= TRAIN_CHANGED | {"batch_size", "accum_steps"}, differ
    assert a["batch_size"] * a["accum_steps"] == b["batch_size"] * b["accum_steps"] == 8
    for k in ("seed", "epochs", "lr", "weight_decay", "amp_dtype", "eval_every_epochs"):
        assert a[k] == b[k], k
    assert b["checkpoint_dir"] != a["checkpoint_dir"]


def test_it_trains_from_scratch_and_says_why_it_deviates(m5, m6):
    """KB Decision 56. Decision 19 mandates resuming from the previous stage's
    best checkpoint; Model D does not, and the deviation must be argued in the
    config rather than left as an unexplained null.

    Two reasons, both recorded there: resuming would give Model D 120 epochs of
    optimisation against Model C's 60, which the M6 gate cannot separate from the
    architecture it measures; and it does not work anyway — AdamW's first
    bias-corrected step is ~lr in magnitude regardless of gradient scale, so at
    lr 2e-4 with no warmup it perturbs C's converged solution rather than
    refining it. A warm restart, not a fine-tune.
    """
    assert m6["train"]["init_from"] is None
    note = m6["train"]["init_from_note"]
    assert "Decision 56" in note, "the deviation from Decision 19 must cite its entry"
    assert "Decision 19" in note, "the rule being deviated from must be named"


def test_training_is_at_parity_with_model_c(m5, m6):
    """The point of training from scratch: the ONLY differences between C and D
    are the temporal fusion and the temporal loss term. If the schedule drifts,
    the gate stops measuring the architecture (KB Decisions 35, 47, 56)."""
    for k in ("seed", "epochs", "lr", "weight_decay", "batch_size", "accum_steps", "amp_dtype"):
        assert m6["train"][k] == m5["train"][k], f"{k} differs from Model C"
    eff = lambda t: int(t["batch_size"]) * int(t.get("accum_steps", 1))
    assert eff(m6["train"]) == eff(m5["train"]) == 8


# --------------------------------------------------------------------------- #
# the §11 pre-cloud gate
# --------------------------------------------------------------------------- #
def test_the_overfit_thresholds_are_model_a_s_unrelaxed(m4, m6):
    """KB §11's local pre-cloud gate is mandatory before the expensive run, and
    Decision 42 fixed these four thresholds before M4 trained. Importing them
    unchanged is what keeps the gate a gate."""
    assert m6["overfit"]["thresholds"] == m4["overfit"]["thresholds"]
    assert m6["overfit"]["n_frames"] == m4["overfit"]["n_frames"] == 32
    assert 32 <= m6["overfit"]["n_frames"] <= 128
    for k in ("max_steps", "batch_size", "lr", "seed"):
        assert m6["overfit"][k] == m4["overfit"][k], k


def test_the_overfit_frame_selection_declares_that_it_is_weaker(m6):
    """Contiguous runs are easier to memorise than scattered frames. No threshold
    moved to compensate, and the direction of the change is stated rather than
    presented as equivalent."""
    o = m6["overfit"]
    assert o["run_length"] == 4
    assert o["n_frames"] % o["run_length"] == 0
    assert "EASIER" in o["n_frames_rationale"].upper()


# --------------------------------------------------------------------------- #
# reports (KB Decision 52)
# --------------------------------------------------------------------------- #
def test_every_report_destination_is_declared_and_m6_specific(m6):
    for key in SUPPORTED_REPORT_KEYS:
        declared = m6["reports"][key]
        assert declared.startswith("eval/reports/m6_temporal")
        assert report_path(m6, key) == ROOT / declared


def test_m6_reports_cannot_collide_with_committed_evidence(m4, m5, m6):
    """Seven of nine config/mode combinations once resolved onto Model A's
    committed artifacts (KB Decision 52). The destinations that matter most here
    are the three M6 reports already committed as the PRE-REGISTERED gate."""
    mine = set(m6["reports"].values()) - {m6["reports"]["notes"]}
    assert not mine & set(LEGACY_REPORTS.values())
    assert not mine & set(m5["reports"].values())
    assert not mine & {
        "eval/reports/m6_temporal_baseline.json",
        "eval/reports/m6_mask_calibration.json",
        "eval/reports/m6_gridsample_probe.json",
    }


def test_the_export_block_claims_nothing_that_was_not_measured(m6):
    """A deployment claim needs a built and executed engine behind it. No Model D
    graph has been exported; the block declares shapes and says so."""
    assert "NOT YET EXERCISED" in m6["export"]["status"]
    assert m6["export"]["opset"] == 17
    assert m6["export"]["dynamic_axes"] is False


def test_train_baseline_refuses_a_temporal_config():
    """The live trap KB Decision 58 closes.

    `TemporalStructFormer.forward` accepts `model(x)` because cold start needs it,
    so `train_baseline --mode eval` on the M6 config would have run Model D
    WITHOUT history — scoring a model that does not exist — and written it to
    `eval/reports/m6_temporal_metrics.json`, a destination the M6 config declares
    and `assert_reports_resolvable` therefore permits.
    """
    import subprocess, sys
    from renderer.training.train_baseline import ROOT

    r = subprocess.run(
        [sys.executable, "-m", "renderer.training.train_baseline",
         "--config", "configs/temporal/m6_temporal.json", "--mode", "eval"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode != 0, "train_baseline accepted a temporal config"
    assert "train_temporal" in (r.stderr + r.stdout), "the refusal must name the right entry point"
