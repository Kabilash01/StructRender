#!/usr/bin/env bash
# M0 exit criterion (c): inject ONE known, realistic fault so the agent
# build-test-repair loop can be demonstrated catching and fixing it.
#
# Fault: a typo in the enemy-destruction call in game_project/scripts/enemy.gd
#   GameState.destroy_enemy(name)  ->  GameState.destroy_enmy(name)
# This is a runtime "nonexistent function" error only hit when the player
# interacts with the enemy, so the game parses fine but can never reach the
# win state. Reverse it with scripts/restore_demo.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/game_project/scripts/enemy.gd"

if grep -q "GameState.destroy_enmy(name)" "$TARGET"; then
  echo "break_demo: fault already present in $TARGET"
  exit 0
fi
if ! grep -q "GameState.destroy_enemy(name)" "$TARGET"; then
  echo "break_demo: expected line 'GameState.destroy_enemy(name)' not found in $TARGET" >&2
  exit 1
fi

sed -i 's/GameState\.destroy_enemy(name)/GameState.destroy_enmy(name)/' "$TARGET"
echo "break_demo: injected fault into $TARGET"
grep -n "destroy_enmy" "$TARGET"
