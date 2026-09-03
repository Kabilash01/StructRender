#!/usr/bin/env bash
# Milestone 2 — dataset recorder verification (GPU, like test_structural.sh).
#
# Records a TINY fixture (1 short temporal window + the {0.5,1,2}s camera_away
# rungs for one entity) into a throwaway temp dataset root and runs the recorder
# assertions (game_project/tests/dataset/d1_recorder.gd). The full seed dataset
# is a separate script: scripts/record_seed_dataset.sh.
#
# Needs a real rendering device (the exporter + native-RGB viewport read pixels
# back). CI without a GPU should skip: STRUCTRENDER_SKIP_STRUCTURAL=1.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME_DIR="$ROOT/game_project"
GODOT="${GODOT:-godot}"
RENDER_DRIVER="${STRUCTRENDER_RENDER_DRIVER:-vulkan}"
# 360 s (not 240): the recorder scene renders ~66 GPU frames + assertions, and
# under combined-suite load (right after the S1-S7 scenes) the 240 s ceiling was
# hit intermittently while the same scene passes comfortably standalone.
TIMEOUT_S="${DATASET_TIMEOUT_S:-360}"

if [[ "${STRUCTRENDER_SKIP_STRUCTURAL:-0}" == "1" ]]; then
  echo "[test_dataset] SKIPPED (STRUCTRENDER_SKIP_STRUCTURAL=1)"
  exit 0
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/structrender_ds_ci.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT

export STRUCTRENDER_DATASET_ROOT="$TMP_ROOT"
export STRUCTRENDER_GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "[test_dataset] recording CI fixture into $TMP_ROOT"
out="$(timeout "$TIMEOUT_S" "$GODOT" --path "$GAME_DIR" --rendering-driver "$RENDER_DRIVER" \
      --script "res://tests/dataset/d1_recorder.gd" 2>&1)"
code=$?
echo "$out" | grep -vE '"event":"(frame_tick|game_start|game_end|sim_driver_start|sim_driver_done)"'

ok=1
if [[ $code -ne 0 ]]; then echo "   -> godot exited $code"; ok=0; fi
if grep -qE "SCRIPT ERROR|Parse Error|Failed to load script|Failed to instantiate" <<<"$out"; then
  echo "   -> script/parse error"; ok=0
fi
if grep -q "HARNESS_RESULT FAIL" <<<"$out"; then ok=0; fi
if ! grep -q "HARNESS_RESULT PASS" <<<"$out"; then echo "   -> no HARNESS_RESULT PASS marker"; ok=0; fi

if [[ $ok -eq 1 ]]; then
  echo "[test_dataset] RESULT: PASS"
else
  echo "[test_dataset] RESULT: FAIL"
  exit 1
fi
