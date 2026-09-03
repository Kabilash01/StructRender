---
name: m0-arena-collector
description: What Milestone 0's "Arena Collector" Godot game is and the design choices M1 will build on
metadata:
  type: project
---

M0 delivered `game_project/` — a minimal 3D Godot game plus a headless
test/repair loop. Built 2026-09-03.

**Why:** M0 is the seed the later milestones instrument; M1 (structural export +
EntityKey) attaches to its entity nodes, so their names and structure are a
contract.

**How to apply:** when starting M1, treat these as fixed unless a KB Decision Log
entry changes them.

- Renderer is **Forward+**, pinned in `project.godot` (`renderer/rendering_method`
  and `.mobile` both `forward_plus`). Physics fixed at 60 tps.
- Autoloads: `GameState` (authoritative state), `GameLog` (NDJSON logger —
  not `Logger`, see [[godot47-headless-testing]]), `SimDriver`.
- Stable entity node names (spec contract): `Player`, `Player/MainCamera`,
  `Coin_0`..`Coin_4` (Area3D), `Enemy_0` (CharacterBody3D, spawned at runtime by
  `Spawner` as a child of the main scene root), `Arena`, `Spawner`.
- `SimDriver` runs only headless (or `structrender_sim` feature); it closed-loop
  steers the Player with real input actions toward each fixed coin then the
  enemy, wins ~tick 508, quits 0 (quits 1 + logs `error` on 3000-tick timeout).
  It is a closed-loop controller over fixed positions, not a blind timed script —
  deterministic because the scene layout has no randomness.
- One command: `./scripts/test.sh` = `check_layout.sh` + `check_launch.sh`
  (end-to-end headless win) + every `game_project/tests/test_*.gd`.
- Repair demo: `scripts/break_demo.sh` / `scripts/restore_demo.sh` (sed a typo
  into `enemy.gd`), transcript in `agent/workflows/repair_demo.md`.
- Not built at M0 (later milestones): any structural buffer, EntityKey, lifecycle
  event schema, dataset recorder, anything under `renderer/` or `inference/`.
