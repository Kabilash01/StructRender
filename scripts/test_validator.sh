#!/usr/bin/env bash
# Milestone 2 — dataset VALIDATOR verification (no GPU needed).
#
# Runs the pytest suite in data_pipeline/validation/tests/. It discovers the
# recorded seed session (the newest ds_* dir under $STRUCTRENDER_DATASET_ROOT,
# default data_pipeline/recorder/, or $STRUCTRENDER_SEED_SESSION), builds
# corrupted sessions programmatically from it, and asserts each fault is
# rejected, plus that a pristine session passes and gets a VALIDATED marker.
#
# The seed session is gitignored — without it the suite cannot verify M2
# exit criterion 1, so this script exits 2 (SKIPPED, distinct from PASS) rather
# than letting an all-skipped pytest run masquerade as a pass. Force-skip in CI
# with STRUCTRENDER_SKIP_VALIDATOR=1. Regenerate the seed with
# scripts/record_seed_dataset.sh, or point at one with $STRUCTRENDER_SEED_SESSION.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${STRUCTRENDER_PY:-$ROOT/.venv/bin/python}"
TIMEOUT_S="${VALIDATOR_TIMEOUT_S:-600}"
SEED_ROOT="${STRUCTRENDER_DATASET_ROOT:-$ROOT/data_pipeline/recorder}"

if [[ "${STRUCTRENDER_SKIP_VALIDATOR:-0}" == "1" ]]; then
  echo "[test_validator] SKIPPED (STRUCTRENDER_SKIP_VALIDATOR=1)"
  exit 2
fi

if [[ ! -x "$PY" ]]; then
  echo "[test_validator] no venv python at $PY — run: python -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi

if [[ -z "${STRUCTRENDER_SEED_SESSION:-}" ]]; then
  shopt -s nullglob
  seed_dirs=("$SEED_ROOT"/ds_*/)
  shopt -u nullglob
  if [[ ${#seed_dirs[@]} -eq 0 ]]; then
    echo "[test_validator] SKIPPED — no seed session under $SEED_ROOT"
    echo "                (run scripts/record_seed_dataset.sh, or set \$STRUCTRENDER_SEED_SESSION)"
    exit 2
  fi
fi

echo "[test_validator] running data_pipeline/validation/tests/"
timeout "$TIMEOUT_S" "$PY" -m pytest -q "$ROOT/data_pipeline/validation/tests/"
code=$?

if [[ $code -eq 0 ]]; then
  echo "[test_validator] RESULT: PASS"
else
  echo "[test_validator] RESULT: FAIL (pytest exit $code)"
  exit 1
fi
