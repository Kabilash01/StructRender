#!/usr/bin/env bash
# The ONE command for M0 exit criterion 2.
#
#   1. layout check           (scripts/check_layout.sh)
#   2. launch / scene-load     (agent/validators/check_launch.sh)
#   3. every headless test in  game_project/tests/test_*.gd
#
# Prints a PASS/FAIL summary and exits non-zero if anything fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME_DIR="$ROOT/game_project"
GODOT="${GODOT:-godot}"
TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-90}"

pass=0
fail=0
skip=0
declare -a results

# A step may exit 2 to mean SKIPPED (e.g. a GPU/fixture-gated step whose inputs
# are absent) — distinct from PASS so the summary never claims a step verified
# something it never ran.
run_step() {
  local name="$1"; shift
  local code
  echo ""
  echo "=================================================================="
  echo ">> $name"
  echo "=================================================================="
  "$@"; code=$?
  if [[ $code -eq 0 ]]; then
    results+=("PASS  $name")
    pass=$((pass + 1))
  elif [[ $code -eq 2 ]]; then
    results+=("SKIP  $name")
    skip=$((skip + 1))
  else
    results+=("FAIL  $name")
    fail=$((fail + 1))
  fi
}

run_gdscript_test() {
  local script_path="$1"
  local out code
  out="$(timeout "$TEST_TIMEOUT_S" "$GODOT" --headless --path "$GAME_DIR" --script "$script_path" 2>&1)"
  code=$?
  echo "$out"
  if [[ $code -ne 0 ]]; then
    echo "   -> godot exited $code"
    return 1
  fi
  if grep -qE "SCRIPT ERROR|Parse Error|Failed to load script" <<<"$out"; then
    echo "   -> script/parse error"
    return 1
  fi
  if grep -q "HARNESS_RESULT FAIL" <<<"$out"; then
    return 1
  fi
  if ! grep -q "HARNESS_RESULT PASS" <<<"$out"; then
    echo "   -> no HARNESS_RESULT marker (test did not complete)"
    return 1
  fi
  return 0
}

run_step "layout check" "$ROOT/scripts/check_layout.sh"
run_step "launch / scene-load gate" "$ROOT/agent/validators/check_launch.sh"

shopt -s nullglob
for t in "$GAME_DIR"/tests/test_*.gd; do
  run_step "gdscript test: $(basename "$t")" run_gdscript_test "res://tests/$(basename "$t")"
done
shopt -u nullglob

# M1 structural-export verification scenes S1-S7 (need a real GPU; see the script).
run_step "structural export S1-S7" "$ROOT/scripts/test_structural.sh"

# M2 dataset recorder — tiny fixture + recorder assertions (needs a real GPU).
run_step "dataset recorder (M2)" "$ROOT/scripts/test_dataset.sh"

# M2 dataset validator — pytest suite over the seed session (no GPU; reports
# SKIP, not PASS, if the gitignored seed session is absent).
run_step "dataset validator (M2)" "$ROOT/scripts/test_validator.sh"

# M3 offline teacher targets — adapter contract, style-by-config, cache and
# alignment validation (no GPU / no model download; SKIP without the seed session).
run_step "teacher targets (M3)" "$ROOT/scripts/test_teacher.sh"

# M4 CNN/U-Net baseline — split contract, input channels, overfit thresholds and
# ONNX export parity (the overfit check needs CUDA and skips without it; the whole
# step SKIPs without the seed session / targets).
run_step "cnn baseline (M4)" "$ROOT/scripts/test_baseline.sh"

# M6 engine-guided temporal reprojection — warp direction/convention, the
# confidence mask's hard zeros, the temporal-error metric, and the assertions
# that the three committed M6 artifacts still say what they are quoted as saying.
# Pure-tensor parts run anywhere; data- and artifact-backed parts SKIP cleanly.
run_step "temporal reprojection (M6)" "$ROOT/scripts/test_temporal.sh"

echo ""
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
for r in "${results[@]}"; do
  echo "  $r"
done
echo "------------------------------------------------------------------"
echo "  $pass passed, $fail failed, $skip skipped"

if [[ $fail -ne 0 ]]; then
  echo "  RESULT: FAIL"
  exit 1
fi
if [[ $skip -ne 0 ]]; then
  echo "  RESULT: PASS (with $skip skipped — not a full verification)"
  exit 0
fi
echo "  RESULT: PASS"
