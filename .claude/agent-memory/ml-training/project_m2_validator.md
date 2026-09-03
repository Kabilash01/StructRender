---
name: m2-validator
description: M2 dataset validator location, design decisions, and how to run it
metadata:
  type: project
---

The Milestone 2 dataset validator was built 2026-09-03 (this agent). It is the
last M2 deliverable; the recorder side was already done.

- Code: `data_pipeline/validation/validator.py` (core), `__main__.py` (CLI:
  `python -m data_pipeline.validation <session_dir>`), `tests/` (pytest, 39 tests).
- Runner: `scripts/test_validator.sh`, wired into `scripts/test.sh` after the
  "dataset recorder (M2)" step. No GPU needed. Skips when the gitignored seed
  session is absent (`STRUCTRENDER_SKIP_VALIDATOR=1` also skips).
- Reference seed session it validates: `data_pipeline/recorder/ds_1788430292_4c77250c`
  (990 frames, gitignored). Passes clean; `VALIDATED` marker written.
- Acceptance rules are pinned in **KB Decision 29** (partial-dropout = strict
  per-pixel channel cross-agreement tol 0; motion magnitude unbounded; temporal
  window first frame motion_valid must be false; reappearance derived fields
  recomputed not trusted; debug-log separation enforced structurally).
- Deps added to `pyproject.toml`: `numpy>=2.0`, `pillow>=10.0` in `dependencies`;
  `pytest>=8.0` in `[project.optional-dependencies] dev`.

**Why:** M2 exit criteria (① corrupted sessions rejected, ② valid sessions
replay) both depend on this; without pinned thresholds the pass/fail verdict is
not reproducible.

**How to apply:** M2 is complete. Next real training work is M3 (teacher
targets) then M4 (CNN/U-Net baseline). If asked to extend the validator, keep it
report-only (never repair/mutate) and add a dated KB Decision Log entry for any
new acceptance rule. See [[feedback-validator-scope]].
