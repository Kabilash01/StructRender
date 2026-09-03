# M0 Game Specification — "Arena Collector"

Status: Milestone 0 deliverable ("minimal 3D game", KB §10).
This spec is deliberately minimal — its only job is to give M0 a 3D game that
**runs**, is **testable with one command**, and can be **broken and repaired**
by the coding-agent loop. It is also the seed the later milestones instrument.

## Purpose

Exercise the coding-agent build-test-repair workflow (KB §3A) on a real, small
Godot project, and leave behind a game with clearly-identified entities that
Milestone 1 can attach structural export and entity-lifecycle hooks to.

## Engine

- Godot 4.x, version pinned in `/.godot-version`.
- Forward+ or Mobile renderer (either is fine for M0; pin the choice in
  `project.godot` and do not change it without a KB Decision Log note).
- Must run both windowed and `--headless`.

## Scene

Single main scene `game_project/scenes/main.tscn`:

- **Arena**: a `StaticBody3D` ground plane (~20×20) with four low perimeter
  walls. Fixed, no movement.
- **Player**: a `CharacterBody3D` capsule named `Player`.
  - Keyboard only (no mouse-look — keeps runs deterministic and testable):
    `move_forward` / `move_back` / `move_left` / `move_right` (mapped to WASD
    and arrow keys in `project.godot`), `interact` (Space).
  - Moves in the XZ plane at a fixed speed; gravity applied; collides with
    walls. No jump.
- **Camera3D**: fixed third-person offset following `Player` position (no
  rotation input). Named `MainCamera`.
- **Coins**: exactly 5 `Area3D` nodes named `Coin_0` … `Coin_4` at fixed
  positions (no randomness). Each: entering its area with `Player` →
  `GameState.collect_coin(name)` then the coin frees itself.
- **Spawner**: an empty `Node3D` named `Spawner`. On `_ready` (frame 1) it
  spawns exactly one enemy.
- **Enemy**: a `CharacterBody3D` named `Enemy_0`, instanced from
  `game_project/scenes/enemy.tscn`. Moves toward `Player` at a fixed speed
  each physics tick (pure pursuit, no pathfinding). If `Player` presses
  `interact` while within 2.0 m of `Enemy_0`, the enemy is destroyed:
  `GameState.destroy_enemy(name)` then the enemy frees itself.

Entity node names (`Player`, `MainCamera`, `Coin_0..4`, `Enemy_0`, `Spawner`)
are **stable and are part of this contract** — M1's EntityKey work depends on
them. Do not rename without updating this spec and the tests.

## Rules / state

`GameState` autoload (singleton, `game_project/scripts/game_state.gd`):

- `score: int` — +1 per coin collected.
- `coins_remaining: int` — starts at 5, decrements on collect.
- `enemy_alive: bool`.
- `elapsed_ticks: int` — incremented each `_physics_process`.
- `won: bool` — set true when `coins_remaining == 0 and not enemy_alive`.
- Signals: `coin_collected(name)`, `enemy_spawned(name)`,
  `enemy_destroyed(name)`, `game_won()`.
- Deterministic: no `randf`/`randi` anywhere in gameplay. If any randomness is
  ever introduced, it must be seeded from a fixed constant.

Win: all coins collected **and** enemy destroyed → `game_won` emitted once.

## Headless / simulation driver

`game_project/scripts/sim_driver.gd` (autoload, active only when the
`--headless` flag or a `structrender_sim` feature tag is present):

- Drives a fixed, scripted sequence of inputs over a bounded number of physics
  ticks that collects all 5 coins and destroys the enemy, then asserts
  `GameState.won == true` and calls `get_tree().quit(0)`.
- If the win state is not reached within a tick budget (e.g. 3000 ticks), it
  logs an error and calls `get_tree().quit(1)`.
- This is what lets `./scripts/test.sh` run the whole game end-to-end without a
  display and get a meaningful exit code.

## Logging

`GameLog` autoload (`game_project/scripts/logger.gd`), KB §8 "Logging &
Observability" style — newline-delimited JSON, one object per line:

> Note: the autoload is named `GameLog`, not `Logger` — Godot 4.7.2 ships a
> built-in `Logger` class that shadows the autoload identifier in GDScript.
> See KB Decision Log #22 (2026-09-03).

- Writes to `user://logs/session_<unix_ts>.jsonl` **and** echoes to stdout.
- Events: `game_start`, `coin_collected`, `enemy_spawned`, `enemy_destroyed`,
  `game_won`, `game_end`. Each line has at least `event`, `tick`, `t_wall_s`.
- One periodic `frame_tick` sample every 60 physics ticks with
  `score`, `coins_remaining`, `enemy_alive`.
- Every line must be valid JSON (a test parses the file back).

## Out of scope for M0 (do NOT build these — they are later milestones)

- No structural buffers: depth / normal / albedo / material / motion /
  entity-ID map (Milestone 1).
- No `EntityKey` (`<session_id>:<instance_id>:<generation>`), no
  spawn/despawn/visibility *events* in the KB's lifecycle-adapter schema
  (Milestone 1). M0 only needs node names to be stable.
- No dataset recorder, manifest, or validator (Milestone 2).
- No renderer, ONNX, TensorRT, C++, EPAM, style conditioning — nothing under
  `renderer/` or `inference/`.
- No GDExtension / C++ build (Milestone 10).
- No mouse-look, no procedural generation, no menus, no audio, no assets
  beyond primitive meshes.

## Acceptance (feeds M0 exit criteria)

1. `godot --headless --path game_project` reaches `game_won` and exits 0.
2. `./scripts/test.sh` runs the full suite and exits 0; exits non-zero when any
   assertion fails.
3. A deliberately broken scene/script (see `agent/workflows/repair_demo.md`) is
   caught by the suite and repaired back to green via the agent loop.
