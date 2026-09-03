---
name: godot47-structural-export
description: Godot 4.7.2 quirks for rendering structural buffers (depth/normal/albedo/id) via SubViewport + override shaders, found building M1
metadata:
  type: project
---

Verified against Godot `4.7.2.stable.official.ed1daf0bf` while building M1
(`game_project/instrumentation/structural_exporter.gd`).

**Why:** the SubViewport + override-shader + `get_image()` readback path is the
whole M1 structural exporter; several behaviours are non-obvious and each cost a
debugging cycle.

**How to apply:** when touching the exporter, the S1-S6 scenes, or the M10
runtime capture path.

- **`--headless` cannot render.** The headless display driver only supports the
  `dummy` rendering driver (`godot --help` confirms), so `get_image()` returns
  blank. The S1-S6 harness runs `godot --path game_project --rendering-driver
  vulkan --script res://tests/structural/sN.gd` (real GPU, brief window). The
  machine has an RTX 5060 + Vulkan + a Wayland/X session. `scripts/test_structural.sh`
  wraps this and is called by `scripts/test.sh`; it honours
  `STRUCTRENDER_SKIP_STRUCTURAL=1` for GPU-less CI. The M0 suite stays headless.
- **RGBA8 viewport applies the sRGB OETF** on readback (shader linear 0.2 comes
  back ~0.485). Set `SubViewport.use_hdr_2d = true` → `get_image()` returns
  `Image.FORMAT_RGBH` (format 14, RGB half-float, **no alpha**) with linear
  values preserved and **no tonemap**, so arbitrary floats round-trip.
- **Write data to `ALBEDO` with `render_mode unshaded`.** `ALBEDO` carries
  arbitrary floats in the HDR target — tested `4.5` and negative values
  (`-1..1` normals) round-trip exactly. `EMISSION` is **ignored** in `unshaded`
  mode. Half-float precision ≈ 1e-4 near 1.0 (fine for normal/albedo with
  documented ε); integers 0..2048 are exact (entity-ID channels encode
  `id & 255` per channel → exact).
- **Background:** a `SubViewport` sharing `world_3d` inherits the scene's
  `WorldEnvironment` (ambient etc.), so background is NOT black. Fix: set a
  `Camera3D.environment` override on the export camera — `BG_COLOR` black,
  `ambient_light_source = DISABLED`, `tonemap_mode = LINEAR`. Then background is
  exactly `(0,0,0)` and the depth pass marks "no geometry" as eye-Z ≤ 0.
- **`Basis` has no `get_column()`** in 4.7 GDScript — use `basis.x / .y / .z`
  (these ARE the columns). `Transform3D * Vector4` is not defined; wrap as
  `Projection(transform) * v4`. `Projection.x/.y/.z/.w` are the columns (Vector4).
- **Render sync:** per pass, set `material_override` on every `MeshInstance3D`,
  then `await get_tree().process_frame` ×2 + `await RenderingServer.frame_post_draw`,
  then `get_texture().get_image()`. One `process_frame` is not enough. Restore
  overrides after all passes.
- **`Camera3D.project_position(screen, z)` / `unproject_position(world)`** work
  from a camera inside a SubViewport (must be `current` + in tree), and `z` is
  **planar eye-space Z** (matches the depth channel), so unproject/reproject
  round-trips exactly — used for the motion computation instead of hand-rolled
  matrix inversion. The stored 4×4 `projection @ view` matched
  `unproject_position` to < 0.01 px in S1.
- **Perf:** `get_image()` on 512×288 ≈ 1 ms; a full per-pixel GDScript loop
  (147 k px) with a matrix-vector multiply ≈ 8 ms. A whole 4-pass export +
  motion + validation is well under 100 ms. Coroutine `await` resumes fine
  under a `SceneTree` `--script` main loop.
- **A GDScript parse error in a `--script` SceneTree file leaves an empty main
  loop that never quits** → the run hangs until the outer `timeout`. Always
  grep stdout for `SCRIPT ERROR` / `Parse Error` AND a success marker (same
  lesson as the M0 headless note, but the failure mode here is a hang not a
  fast exit-0).
