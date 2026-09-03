#!/usr/bin/env bash
# Milestone 1 — structural-export verification scenes S1-S7.
#
# These need a REAL rendering device (GPU): the exporter drives a half-float
# SubViewport and reads pixels back, which the `--headless` dummy driver cannot
# do. So unlike the M0 suite these run with a real display/rendering driver.
# Requires a working GPU + Vulkan (or an X/Wayland session). CI without a GPU
# should skip this script (STRUCTRENDER_SKIP_STRUCTURAL=1).
#
# Called by scripts/test.sh. Exits non-zero if any scene fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME_DIR="$ROOT/game_project"
GODOT="${GODOT:-godot}"
RENDER_DRIVER="${STRUCTRENDER_RENDER_DRIVER:-vulkan}"
TIMEOUT_S="${STRUCTURAL_TIMEOUT_S:-150}"

if [[ "${STRUCTRENDER_SKIP_STRUCTURAL:-0}" == "1" ]]; then
  echo "[test_structural] SKIPPED (STRUCTRENDER_SKIP_STRUCTURAL=1)"
  exit 0
fi

pass=0
fail=0
declare -a results

for t in "$GAME_DIR"/tests/structural/s[0-9]*.gd; do
  name="$(basename "$t")"
  echo ""
  echo "=================================================================="
  echo ">> structural scene: $name"
  echo "=================================================================="
  out="$(timeout "$TIMEOUT_S" "$GODOT" --path "$GAME_DIR" --rendering-driver "$RENDER_DRIVER" \
        --script "res://tests/structural/$name" 2>&1)"
  code=$?
  echo "$out" | grep -vE '"event":"(frame_tick|game_start|game_end)"'
  ok=1
  if [[ $code -ne 0 ]]; then echo "   -> godot exited $code"; ok=0; fi
  if grep -qE "SCRIPT ERROR|Parse Error|Failed to load script|Failed to instantiate" <<<"$out"; then
    echo "   -> script/parse error"; ok=0
  fi
  if grep -q "HARNESS_RESULT FAIL" <<<"$out"; then ok=0; fi
  if ! grep -q "HARNESS_RESULT PASS" <<<"$out"; then
    echo "   -> no HARNESS_RESULT PASS marker"; ok=0
  fi
  if [[ $ok -eq 1 ]]; then results+=("PASS  $name"); pass=$((pass+1)); else results+=("FAIL  $name"); fail=$((fail+1)); fi
done

echo ""
echo "=================================================================="
echo "STRUCTURAL SUMMARY"
echo "=================================================================="
for r in "${results[@]}"; do echo "  $r"; done
echo "------------------------------------------------------------------"
echo "  $pass passed, $fail failed"
if [[ $fail -ne 0 ]]; then echo "  RESULT: FAIL"; exit 1; fi
echo "  RESULT: PASS"
