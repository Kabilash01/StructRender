#!/usr/bin/env bash
# Launch / scene-load gate (KB §3A step 4, build_test_repair.md step 4).
#
# Runs the game end-to-end, headless, driven by sim_driver.gd. Passes only if
# Godot exits 0, the run reaches the win state (`game_won` event), and no script
# / parse errors were printed. Non-zero exit on any failure.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GAME_DIR="$ROOT/game_project"
GODOT="${GODOT:-godot}"
TIMEOUT_S="${LAUNCH_TIMEOUT_S:-120}"

echo "[check_launch] godot --headless --path $GAME_DIR"
out="$(timeout "$TIMEOUT_S" "$GODOT" --headless --path "$GAME_DIR" 2>&1)"
code=$?
echo "$out"

fail=0
if [[ $code -ne 0 ]]; then
  echo "[check_launch] FAIL: godot exited $code"
  fail=1
fi
if grep -qE "SCRIPT ERROR|Parse Error|Failed to load script|Failed to instantiate" <<<"$out"; then
  echo "[check_launch] FAIL: script/parse error in output"
  fail=1
fi
if ! grep -q '"event":"game_won"' <<<"$out"; then
  echo "[check_launch] FAIL: never reached game_won"
  fail=1
fi

if [[ $fail -ne 0 ]]; then
  echo "[check_launch] RESULT: FAIL"
  exit 1
fi
echo "[check_launch] RESULT: PASS (reached game_won, exit 0)"
