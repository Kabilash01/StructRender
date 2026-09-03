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
declare -a results

run_step() {
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo ">> $name"
  echo "=================================================================="
  if "$@"; then
    results+=("PASS  $name")
    pass=$((pass + 1))
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

echo ""
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
for r in "${results[@]}"; do
  echo "  $r"
done
echo "------------------------------------------------------------------"
echo "  $pass passed, $fail failed"

if [[ $fail -ne 0 ]]; then
  echo "  RESULT: FAIL"
  exit 1
fi
echo "  RESULT: PASS"
