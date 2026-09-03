---
name: godot47-dataset-recording
description: Godot 4.7.2 quirks hit building the M2 dataset recorder — .npz via ZIPPacker, half-float, native-RGB SubViewport vs the exporter's material swaps
metadata:
  type: project
---

Verified against Godot `4.7.2.stable.official.ed1daf0bf` building M2
(`game_project/instrumentation/dataset_recorder.gd` / `npz_writer.gd`).

**Why:** each of these cost a debugging cycle and is non-obvious.

**How to apply:** when touching the recorder, npz writer, or any new capture path.

- **`.npz` = ZIP of `.npy` members.** `ZIPPacker` (`open` / `start_file` /
  `write_file` / `close_file` / `close`) writes a valid npz that Python
  `zipfile` + numpy read. `ZIPPacker.set_compression_level(6)` exists in 4.7 and
  gives deflate. Member must be named `<key>.npy` for `np.load(z)["<key>"]`.
  `get_method_list()` on `ZIPPacker` does **not** list its own methods (only
  inherited Object ones) — don't probe it that way, just call them.
- **`ZIPPacker` embeds a per-file modification time** → two runs with identical
  buffer content produce **non-byte-identical `.npz` containers**. Compare
  *decoded members* (`ZIPReader.read_file`) for determinism tests, not the zip
  bytes. The `.npy` member bytes themselves ARE bit-exact reproducible.
- **Half-float:** no `PackedFloat16Array`. Use `PackedByteArray.encode_half(off,
  f)` in a loop (`decode_half` to read). Handles `+inf` / `NaN` correctly
  (round-trips to half inf). ~30 ms for the full 5-channel 512×288 frame — fine
  offline.
- **Native RGB must be captured BEFORE calling the M1 exporter's
  `export_frame()`**, not after. The exporter swaps `material_override` on every
  `MeshInstance3D` in the shared `world_3d` for its passes and only restores at
  the very end; a native-RGB `SubViewport` sharing that world sees the override
  shaders during the ~8 frames `export_frame` awaits, and the restore doesn't
  reliably settle before a post-export readback (intermittent black / stale
  frames, non-deterministic on later frames). Capturing first, while materials
  are pristine, is clean and deterministic.
- **Native-RGB SubViewport:** `use_hdr_2d = false` (RGBA8) is what you want here —
  the sRGB OETF on readback gives a correct 8-bit sRGB `.png`. Share
  `world_3d` so it picks up the scene's `WorldEnvironment` + lights; a plain
  shared-world viewport renders lit and stable within ~1 frame (unlike the
  exporter's override path which needs 2+). Give the scene an actual
  `DirectionalLight3D` + `WorldEnvironment` or the native render is near-black.
- **Autoload `_ready` runs AFTER a `--script` SceneTree's `_initialize`** starts.
  `GameLog.get_log_path()` is `""` until you `await process_frame` a few times.
  Also `get_log_path()` returns the raw `user://...` path — globalize before
  comparing to a globalized dir.
- **`Node3D.get_world_3d()` errors ("!is_inside_world()")** if called the same
  frame the node was `add_child`-ed under a headless `--script` run; `await
  process_frame` first. (The recorder's native viewport setup trips this
  headless — harmless there since headless can't record anyway.)
- `String.sha256_text()`, `String.get_slice(multichar_delim, idx)`,
  `String.humanize_size()`, `absi/absf`, `RegEx` all work as expected in 4.7.
- See [[godot47-structural-export]] for the exporter-side rendering quirks.
