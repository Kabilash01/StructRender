#!/usr/bin/env bash
# Milestone 2 — regenerate the seed dataset (a test fixture; NOT committed —
# .gitignore covers data_pipeline/recorder/*). NOT part of scripts/test.sh; the
# tiny CI fixture is scripts/test_dataset.sh.
#
# Needs a real rendering device (GPU). Takes several minutes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME_DIR="$ROOT/game_project"
GODOT="${GODOT:-godot}"
RENDER_DRIVER="${STRUCTRENDER_RENDER_DRIVER:-vulkan}"
# ~990 frames, each stepping the sim ~8x with a full material swap (KB Decision
# 26) plus a native-RGB viewport read — budget ~1 s/frame under load.
TIMEOUT_S="${SEED_TIMEOUT_S:-1800}"

export STRUCTRENDER_DATASET_ROOT="${STRUCTRENDER_DATASET_ROOT:-$ROOT/data_pipeline/recorder}"
export STRUCTRENDER_GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
mkdir -p "$STRUCTRENDER_DATASET_ROOT"

echo "[record_seed] writing to $STRUCTRENDER_DATASET_ROOT"
out="$(timeout "$TIMEOUT_S" "$GODOT" --path "$GAME_DIR" --rendering-driver "$RENDER_DRIVER" \
      --script "res://tests/dataset/seed_record.gd" 2>&1)"
code=$?
echo "$out" | grep -vE '"event":"(frame_tick|game_start|game_end|sim_driver_start|sim_driver_done)"'

if [[ $code -ne 0 ]] || grep -qE "SCRIPT ERROR|Parse Error" <<<"$out" || ! grep -q "HARNESS_RESULT PASS" <<<"$out"; then
  echo "[record_seed] RESULT: FAIL"
  exit 1
fi
echo "[record_seed] RESULT: PASS"
