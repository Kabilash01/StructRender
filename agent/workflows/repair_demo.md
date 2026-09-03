# M0 Repair Demo — build-test-repair loop on a real fault

Demonstrates M0 exit criterion (c): a deliberately broken script is caught by
`./scripts/test.sh` and repaired back to green via the agent loop
(`agent/workflows/build_test_repair.md`, steps 5–7).

Reproduce:

```
./scripts/break_demo.sh      # inject the fault
./scripts/test.sh            # RESULT: FAIL
./scripts/restore_demo.sh    # revert the fault
./scripts/test.sh            # RESULT: PASS
```

---

## 1. The fault

`scripts/break_demo.sh` rewrites one line in `game_project/scripts/enemy.gd`:

```
-        GameState.destroy_enemy(name)
+        GameState.destroy_enmy(name)
```

A realistic typo: the method name is misspelled. `game_state.gd` is an untyped
`Node`, so GDScript does not catch this at parse time — the script loads fine and
the game boots. The error only fires at runtime, on the physics tick where the
player interacts with `Enemy_0`.

## 2. Failure output (`./scripts/test.sh`)

```
[check_launch] godot --headless --path /home/kabilash01/StructRender/game_project
...
{"coins_remaining":0,"event":"coin_collected","name":"Coin_4","score":5,"tick":489}
SCRIPT ERROR: Invalid call. Nonexistent function 'destroy_enmy' in base 'Node (game_state.gd)'.
          at: _physics_process (res://scripts/enemy.gd:42)
          GDScript backtrace (most recent call first):
              [0] _physics_process (res://scripts/enemy.gd:42)
... (repeats every physics tick while the player stays in interact range) ...
{"coins_remaining":0,"enemy_alive":true,"event":"error","message":"win state not reached within tick budget","tick":3001,"where":"sim_driver"}
[check_launch] FAIL: godot exited 1
[check_launch] FAIL: script/parse error in output
[check_launch] FAIL: never reached game_won
[check_launch] RESULT: FAIL
...
>> gdscript test: test_enemy_destroy.gd
  FAIL - GameState.enemy_alive false after destroy (expected false, got true)
  FAIL - Enemy_0 freed itself after destroy
HARNESS_RESULT FAIL (4 checks, 2 failure(s))
...
SUMMARY
  PASS  layout check
  FAIL  launch / scene-load gate
  PASS  gdscript test: test_coin_collection.gd
  FAIL  gdscript test: test_enemy_destroy.gd
  PASS  gdscript test: test_logger_output.gd
  PASS  gdscript test: test_main_scene.gd
  PASS  gdscript test: test_player_movement.gd
  PASS  gdscript test: test_project_loads.gd
  PASS  gdscript test: test_win_condition.gd
  7 passed, 2 failed
  RESULT: FAIL
```

## 3. Diagnosis (INSPECT, build_test_repair.md step 5)

Reading in order:

- **`test.sh` stdout** — two failures: the launch gate and `test_enemy_destroy.gd`.
  `test_coin_collection` still passes, so coin logic and `GameState` wiring are
  fine; the failure is specific to enemy destruction.
- **Godot stderr** — `SCRIPT ERROR: Invalid call. Nonexistent function
  'destroy_enmy' in base 'Node (game_state.gd)'` with `at: _physics_process
  (res://scripts/enemy.gd:42)`. This names the file, line, bad symbol, and
  receiver exactly.
- **`user://logs/session_*.jsonl`** — `enemy_spawned` is present but
  `enemy_destroyed` / `game_won` never appear; the final line is
  `{"event":"error","message":"win state not reached within tick budget", ...}`
  from `sim_driver.gd`. Confirms: the enemy is never destroyed, so the win
  condition can't be met.

Root cause: single misspelled method name at `enemy.gd:42`. `GameState` has
`destroy_enemy`, not `destroy_enmy`.

## 4. The fix (REPAIR — root cause only, no adjacent changes)

```diff
--- a/game_project/scripts/enemy.gd
+++ b/game_project/scripts/enemy.gd
@@
 	if dist <= INTERACT_RANGE and Input.is_action_pressed("interact"):
-		GameState.destroy_enmy(name)
+		GameState.destroy_enemy(name)
 		queue_free()
```

(`./scripts/restore_demo.sh` applies exactly this.)

## 5. Green rerun (`./scripts/test.sh`)

```
{"event":"enemy_destroyed","name":"Enemy_0","tick":508}
{"event":"game_won","score":5,"tick":508,"ticks":508}
...
SUMMARY
  PASS  layout check
  PASS  launch / scene-load gate
  PASS  gdscript test: test_coin_collection.gd
  PASS  gdscript test: test_enemy_destroy.gd
  PASS  gdscript test: test_logger_output.gd
  PASS  gdscript test: test_main_scene.gd
  PASS  gdscript test: test_player_movement.gd
  PASS  gdscript test: test_project_loads.gd
  PASS  gdscript test: test_win_condition.gd
  9 passed, 0 failed
  RESULT: PASS
```

Loop complete: fault injected → suite failed loudly with a file:line diagnostic →
root cause identified from test output + Godot stderr + session log → minimal
one-line fix → suite green.
