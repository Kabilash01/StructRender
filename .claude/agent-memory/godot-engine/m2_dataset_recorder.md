---
name: m2-dataset-recorder
description: What Milestone 2's recorder side delivered (GDScript recorder, npz storage, scenario scripts, seed dataset) and its contracts
metadata:
  type: project
---

M2 delivered the **recorder side** of "Dataset Recorder + Validator". Built
2026-09-03 on M0+M1. The Python **validator + replay checker is a separate task
(ml-training subagent)** — not in this repo yet. Full suite `./scripts/test.sh`
green (adds the headless recorder test + the GPU `test_dataset.sh` fixture).

**Why:** M2 is the frozen, versioned dataset every training milestone (M4+)
consumes read-only. Its manifest + storage layout is a contract.

**How to apply:** treat these as fixed unless a KB Decision Log entry changes them.

- **Contract doc:** `data_pipeline/schemas/dataset_manifest.md` is authoritative
  for the `TrainingFrame` JSONL record, `manifest.json` header, `sequences[]`
  metadata, the on-disk layout, and the `schema_version` bump policy. M1
  `frame_structure.md` still governs channel *values/ranges/conventions* —
  unchanged.
- **Code:** `game_project/instrumentation/` — `dataset_recorder.gd` (a Node, not
  an autoload; a scenario script adds it and calls `configure` → `begin` →
  `await record_frame(tick)` ×N → `add_*_sequence` → `finish`), `npz_writer.gd`
  (npy-blob builder + `ZIPPacker` wrapper).
- **Storage:** one compressed `.npz` per frame (ZIP of `.npy`), members
  `depth/normal/albedo/motion` = `float16` (`<f2`, depth keeps `+inf`),
  `entity_id` = `int32` (`<i4`). Native RGB = 8-bit sRGB `.png`. Per-frame
  `camera.json` sidecar. Manifest paths use `"<npz>#<member>"` URI form.
- **Dataset root:** repo-gitignored `data_pipeline/recorder/` (env
  `STRUCTRENDER_DATASET_ROOT` overrides; falls back to `user://datasets/`).
  Session id `ds_<unix_ts>_<8hex>` — distinct namespace from `GameLog`'s
  `session_<ts>` (never share files, never write under `user://logs/`).
- **Immutability:** `begin()` returns `false` (writes nothing) if the session dir
  already has `manifest.jsonl` / non-empty `frames/`, or if a `VALIDATED` marker
  is present. The recorder only *checks* for `VALIDATED` — the validator writes it.
- **Fail-loud:** a failed exporter frame aborts `record_frame` (`assert`), never
  zero-fills.
- **Scenarios:** `game_project/tests/dataset/` — `dataset_harness.gd` (scene
  build + `run_rung` reappearance-ladder driver), `d1_recorder.gd` (GPU
  verification, run by `scripts/test_dataset.sh`, ~28 s), `seed_record.gd` (full
  seed, run by `scripts/record_seed_dataset.sh` — NOT in `test.sh`).
  `game_project/tests/test_dataset_recorder.gd` (headless: namespace +
  immutability + log/manifest separation, in the M0 suite).
- **Reappearance semantics** are *derived from the recorded frame log*
  (`last_seen_frame` / `first_seen_again_frame` and `unseen_frames` /
  `unseen_seconds` from the boundary frames' `tick`/`timestamp_s`), never
  asserted — stays exact even when the 4 s/8 s gaps are recorded at reduced rate.
  Ladder sweeps {0.5,1,2,4,8} s × {camera_away, entity_away}, 1 target entity.
- **NOT built at M2:** the validator / replay checker (ml-training); styled
  targets (M3 — every M2 frame is `style_id:"native"`, `target_rgb_path:null`);
  the EntityKey *string service* (M7 — `entity_key` here is a bare
  `"%s:%s:%s"` format of the M1 adapter triple).
- See [[godot47-dataset-recording]] for the engine-level quirks hit building it.
