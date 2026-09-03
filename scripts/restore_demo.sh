#!/usr/bin/env bash
# Reverse the fault injected by scripts/break_demo.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/game_project/scripts/enemy.gd"

if grep -q "GameState.destroy_enemy(name)" "$TARGET" && ! grep -q "GameState.destroy_enmy(name)" "$TARGET"; then
  echo "restore_demo: nothing to restore, $TARGET is already good"
  exit 0
fi

sed -i 's/GameState\.destroy_enmy(name)/GameState.destroy_enemy(name)/' "$TARGET"
echo "restore_demo: restored $TARGET"
grep -n "destroy_enemy" "$TARGET"
