---
name: godot47-headless-testing
description: Godot 4.7.2 behaviors that matter when writing headless GDScript tests and sim drivers for this repo
metadata:
  type: project
---

Verified against Godot `4.7.2.stable.official.ed1daf0bf` while building M0.

**Why:** the M0/M1 test harness runs entirely headless via `godot --headless
--path game_project --script res://tests/<t>.gd`; several engine behaviors are
non-obvious and cost real debugging time.

**How to apply:** when writing or debugging headless tests / sim drivers.

- Headless detection: `DisplayServer.get_name() == "headless"` is true.
  `OS.has_feature("headless")` is **false** — do not use it.
- A native `Logger` class exists (`ClassDB.class_exists("Logger")`), so an
  autoload named `Logger` is unusable from GDScript. This repo's logging autoload
  is `GameLog`. See KB Decision Log #22.
- Under `--script` (a `SceneTree`/`MainLoop` script): autoloads **are**
  instantiated (they appear as children of `root`), but their global identifiers
  (`GameState`, `GameLog`, …) do **not** resolve at compile time in that script —
  access them with `root.get_node("GameState")`.
- `--script` with a GDScript parse/compile error still exits **0**. Test runners
  must grep stdout for `SCRIPT ERROR` / `Parse Error` / `Failed to load script`
  AND for an explicit success marker; never trust the exit code alone.
- Global `class_name` classes are not registered when the project has never been
  opened in the editor (no `global_script_class_cache.cfg`). Test helpers use
  `const H := preload("res://tests/harness.gd")` instead of `class_name`.
- Simulated input works headless: `Input.action_press/release("action")` then
  read via `Input.is_action_pressed` / `Input.get_vector` on the next frame.
- A node with `_physics_process` defined gets physics processing re-enabled when
  circumstances change; `set_physics_process(false)` from outside is not
  reliably sticky. Prefer a per-frame guard (the SimDriver checks
  `Engine.has_meta(&"structrender_no_sim")`, which the test harness sets).
- Await frames in a `SceneTree` script with `await self.physics_frame` /
  `await self.process_frame`.
- Spawning nodes from a parent's `_ready` fails with "busy setting up children";
  use `add_child.call_deferred()` or a deferred spawn method.
