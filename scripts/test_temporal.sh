#!/usr/bin/env bash
# Milestone 6 — ENGINE-GUIDED TEMPORAL REPROJECTION verification (steps 0-3).
#
# Runs the pytest suites that cover M6's delivered pieces:
#   renderer/temporal/tests/     — KB §11's required "motion warp direction",
#                                  "camera motion convention" and "confidence mask"
#                                  unit tests, plus the assertions that the three
#                                  committed M6 artifacts still say what the M6
#                                  write-up quotes them as saying (KB Decision 44)
#   eval/metrics/tests/test_temporal.py
#                                — the temporal-error metric, including the
#                                  measured demonstration that blur LOWERS TE
#                                  (the confound that forbids reporting TE alone)
#
# Most of it is pure-tensor and runs on a clean checkout. The data-backed warp
# checks need the gitignored structural session and SKIP with a reason without
# it; the artifact checks SKIP when their report has not been generated (that
# needs CUDA + TensorRT).
#
# Nothing here trains, and nothing here writes to eval/reports/ — the three M6
# artifacts are regenerated only by their own generators:
#   python -m scripts.probe_m6_gridsample          -> m6_gridsample_probe.json
#   python -m scripts.calibrate_m6_mask            -> m6_mask_calibration.json
#   python -m eval.protocols.run_temporal_baseline -> m6_temporal_baseline.json
#
# Force-skip in CI with STRUCTRENDER_SKIP_TEMPORAL=1.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${STRUCTRENDER_PY:-$ROOT/.venv/bin/python}"
TIMEOUT_S="${TEMPORAL_TIMEOUT_S:-900}"

if [[ "${STRUCTRENDER_SKIP_TEMPORAL:-0}" == "1" ]]; then
  echo "[test_temporal] SKIPPED (STRUCTRENDER_SKIP_TEMPORAL=1)"
  exit 2
fi

if [[ ! -x "$PY" ]]; then
  echo "[test_temporal] no venv python at $PY — run: python -m venv .venv && .venv/bin/pip install -e '.[dev,train,export]'"
  exit 1
fi

if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "[test_temporal] SKIPPED — torch not installed (pip install -e '.[train]')"
  exit 2
fi

echo "[test_temporal] running renderer/temporal/tests/ and eval/metrics/tests/test_temporal.py"
timeout "$TIMEOUT_S" "$PY" -m pytest -q \
  "$ROOT/renderer/temporal/tests/" \
  "$ROOT/eval/metrics/tests/test_temporal.py"
code=$?

if [[ $code -eq 0 ]]; then
  echo "[test_temporal] RESULT: PASS"
else
  echo "[test_temporal] RESULT: FAIL (pytest exit $code)"
  exit 1
fi
