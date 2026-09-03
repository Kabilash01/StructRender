---
name: m1-structural-export
description: What Milestone 1 (structural export + entity lifecycle) delivered on top of M0, and the contracts it established
metadata:
  type: project
---

M1 delivered the Godot-side structural exporter + entity lifecycle adapter.
Built 2026-09-03 on top of M0's `game_project/`. Full suite (`./scripts/test.sh`)
green: M0 headless tests + S1-S7 GPU scenes.

**2026-09-03 spec-guardian follow-up:** fixed a latent bug where an entity first
seen after frame 0 (no `t-1` transform in `_prev_entity_xforms`) got garbage
motion from `M_{t-1}=identity`. Now: entity-owned pixel with no prior transform
=> `motion = (0,0)`, same as frame 0 / background. Also pinned background
(`depth==+inf`) `motion=(0,0)` as an explicit choice in schema §4. New scene
`s7_new_entity_motion.gd` guards both (spawns entity B at frame 1, asserts its
pixels are exactly 0, then moving B at frame 2 recovers analytic motion). Both
conventions to be mirrored into KB Decision 25 by the user.

**Why:** M1 is the deterministic data source every later milestone consumes
(M2 recorder, M6 warp, M7 masks/EntityKey). Its exported values + conventions
are a contract.

**How to apply:** treat these as fixed unless a KB Decision Log entry changes them.

- **Contract doc:** `data_pipeline/schemas/frame_structure.md` is authoritative
  for channel shapes, ranges, the `FrameStructure` dict, `EntityEvent`, and the
  motion convention. KB **Decision 25** pins motion = backward / screen-UV /
  Y-down / camera+object / analytic.
- **Code:** `game_project/instrumentation/` — `structural_exporter.gd` (the
  SubViewport multi-pass renderer, not an autoload — a test/recorder adds it and
  calls `await export_frame(id)`), `entity_lifecycle_adapter.gd`,
  `structural_shaders.gd`, `npy_writer.gd`.
- **Entity registration:** nodes in the Godot group `structrender_entity`.
  M0 scene edits added `Player`, `Coin_0..4` (main.tscn `groups=[...]`) and
  `Enemy_0` (spawner.gd `add_to_group`). `Arena` / `Spawner` / `MainCamera` are
  NOT entities. `test_entity_group.gd` (headless, in the M0 suite) guards this.
- **instance_id** is the adapter's own monotonic counter (starts at 1), NOT
  `Node.get_instance_id()`. `generation` bumps + a fresh instance_id is minted
  on engine-index reuse. Test seam: `adapter.engine_index_provider` (Callable).
  `adapter.frozen` makes `scan()` a no-op (S6b seam).
- **No silent failure:** `exporter._validate()` aborts the frame (returns `{}`,
  sets `last_error` naming the channel) on shape/dtype/finite/unit-normal/
  unregistered-id violations. `exporter.debug_disable_pass` simulates a dead
  GPU pass (S6a).
- **Debug dumps:** `exporter.dump_frame(fs, dir)` → per-channel `.npy` + a
  `camera.json`. `scripts/visualize_frame.py` (pure stdlib, no numpy/PIL) turns
  a dump dir into preview PNGs. NPYs land under the Godot `user://` dir
  (`~/.local/share/godot/app_userdata/StructRender Arena Collector (M0)/structural_debug/`).
- **NOT built at M1** (later milestones): dataset recorder / manifest / JSONL
  frame store / validator / native-RGB capture (M2); EntityKey *string* service,
  masks, feature pooling, EPAM (M7/M8); temporal warp consumption (M6).
- See [[godot47-structural-export]] for the engine-level rendering quirks.
