#!/usr/bin/env bash
# Milestone 4 — CNN/U-Net BASELINE verification (ladder Model A).
#
# Runs the pytest suite in renderer/training/tests/, which covers all four M4
# exit criteria:
#   1. model trains                  — asserted against eval/reports/m4_train.json
#   2. model overfits a tiny set     — 32 frames (KB §11's mandated 32-128 floor) /
#                                      2000 steps / batch 2, against the thresholds
#                                      fixed in configs/training/m4_unet_baseline.json
#   3. recognizable styled output    — prediction must beat the unstyled engine frame
#                                      against the teacher target, travel off the native
#                                      palette (not a passthrough), and keep the target's
#                                      high-frequency energy (not a blur)
#   4. export works                  — PyTorch -> ONNX -> onnxruntime parity at fixed shapes
# plus KB §11's required "finite loss" / "gradient flow" unit tests (pure CPU,
# synthetic tensors — these run even on a clean checkout), the split contract
# (train/val disjoint AND non-adjacent) and the 7-channel input contract (no
# motion, no entity IDs).
#
# GPU: the overfit check needs CUDA and skips itself without it. Everything else
# runs on CPU. Nothing here writes to the structural session or the target set,
# and by default nothing overwrites the committed reports under eval/reports/ —
# set STRUCTRENDER_REFRESH_REPORTS=1 to refresh those committed artifacts.
#
# The structural session, the teacher targets and checkpoints/ are all gitignored,
# so a clean checkout cannot verify M4 — this script exits 2 (SKIPPED, distinct
# from PASS) rather than letting an all-skipped pytest run look like verification.
# Force-skip in CI with STRUCTRENDER_SKIP_BASELINE=1.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${STRUCTRENDER_PY:-$ROOT/.venv/bin/python}"
TIMEOUT_S="${BASELINE_TIMEOUT_S:-1800}"
SEED_ROOT="${STRUCTRENDER_DATASET_ROOT:-$ROOT/data_pipeline/recorder}"

if [[ "${STRUCTRENDER_SKIP_BASELINE:-0}" == "1" ]]; then
  echo "[test_baseline] SKIPPED (STRUCTRENDER_SKIP_BASELINE=1)"
  exit 2
fi

if [[ ! -x "$PY" ]]; then
  echo "[test_baseline] no venv python at $PY — run: python -m venv .venv && .venv/bin/pip install -e '.[dev,train,export]'"
  exit 1
fi

if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "[test_baseline] SKIPPED — torch not installed (pip install -e '.[train]')"
  exit 2
fi

if [[ -z "${STRUCTRENDER_SEED_SESSION:-}" ]]; then
  shopt -s nullglob
  seed_dirs=("$SEED_ROOT"/ds_*/)
  shopt -u nullglob
  if [[ ${#seed_dirs[@]} -eq 0 ]]; then
    echo "[test_baseline] SKIPPED — no seed session under $SEED_ROOT"
    echo "                (run scripts/record_seed_dataset.sh, then generate teacher targets)"
    exit 2
  fi
fi

echo "[test_baseline] running renderer/training/tests/"
timeout "$TIMEOUT_S" "$PY" -m pytest -q "$ROOT/renderer/training/tests/"
code=$?

if [[ $code -eq 0 ]]; then
  echo "[test_baseline] RESULT: PASS"
else
  echo "[test_baseline] RESULT: FAIL (pytest exit $code)"
  exit 1
fi
