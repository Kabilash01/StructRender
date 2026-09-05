#!/usr/bin/env bash
# Milestone 3 — offline TEACHER TARGETS verification.
#
# GPU/model use: every check below runs on CPU with the dependency-free
# `deterministic` teacher. The ONE exception is the real-teacher generation test,
# which runs only if torch/diffusers, the pinned SD1.5+ControlNet weights AND
# enough free VRAM are all present — otherwise it skips with a reason. So this
# script needs no GPU and downloads nothing on a clean checkout, but it will
# exercise the GPU on a machine that has already fetched the weights.
#
# Runs the pytest suite in data_pipeline/teacher/tests/. It discovers the
# recorded seed session (newest ds_* under $STRUCTRENDER_DATASET_ROOT, default
# data_pipeline/recorder/, or $STRUCTRENDER_SEED_SESSION), generates targets with
# the dependency-free `deterministic` teacher (KB Decision 31), then asserts:
#   - the adapter contract (shape/dtype/frame-id, bit-exactness, --dry-run);
#   - a second style can be added by CONFIG ALONE, no code change;
#   - the cache (second run generates zero frames, resume after interrupt);
#   - the alignment validator accepts real targets and REJECTS synthetically
#     misaligned ones (shift / rotate / frame swap / hallucinated object).
#
# The diffusion teacher's own generation test skips itself when torch/diffusers
# or the pinned weights are absent — it is never required for this script to pass.
#
# The seed session is gitignored — without it the suite cannot verify the M3
# exit criterion, so this script exits 2 (SKIPPED, distinct from PASS) rather
# than letting an all-skipped pytest run masquerade as verification. Force-skip
# in CI with STRUCTRENDER_SKIP_TEACHER=1. Regenerate the seed with
# scripts/record_seed_dataset.sh, or point at one with $STRUCTRENDER_SEED_SESSION.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${STRUCTRENDER_PY:-$ROOT/.venv/bin/python}"
TIMEOUT_S="${TEACHER_TIMEOUT_S:-900}"
SEED_ROOT="${STRUCTRENDER_DATASET_ROOT:-$ROOT/data_pipeline/recorder}"

if [[ "${STRUCTRENDER_SKIP_TEACHER:-0}" == "1" ]]; then
  echo "[test_teacher] SKIPPED (STRUCTRENDER_SKIP_TEACHER=1)"
  exit 2
fi

if [[ ! -x "$PY" ]]; then
  echo "[test_teacher] no venv python at $PY — run: python -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi

if [[ -z "${STRUCTRENDER_SEED_SESSION:-}" ]]; then
  shopt -s nullglob
  seed_dirs=("$SEED_ROOT"/ds_*/)
  shopt -u nullglob
  if [[ ${#seed_dirs[@]} -eq 0 ]]; then
    echo "[test_teacher] SKIPPED — no seed session under $SEED_ROOT"
    echo "               (run scripts/record_seed_dataset.sh, or set \$STRUCTRENDER_SEED_SESSION)"
    exit 2
  fi
fi

echo "[test_teacher] running data_pipeline/teacher/tests/"
timeout "$TIMEOUT_S" "$PY" -m pytest -q "$ROOT/data_pipeline/teacher/tests/"
code=$?

if [[ $code -eq 0 ]]; then
  echo "[test_teacher] RESULT: PASS"
else
  echo "[test_teacher] RESULT: FAIL (pytest exit $code)"
  exit 1
fi
