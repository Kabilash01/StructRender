# Dataset Manifest Contract — Milestone 2

Authoritative schema for a **recorded dataset session** produced by the Godot
dataset recorder (`game_project/instrumentation/dataset_recorder.gd`). Mirrors
KB §5 *Training Frame* / *Persistence Layer* / *State Lifecycle → Dataset* and
KB §4 *Dataset Recorder*.

This file is the M2 contract for the **recorder side only**. The **validator**
(replay checker, channel/sequence checks) is a separate task (ml-training
subagent); where this doc says "the validator checks X", that is context, not a
recorder responsibility.

The per-frame structural **values, ranges, and conventions** are unchanged from
Milestone 1 — `data_pipeline/schemas/frame_structure.md` still governs depth
semantics (`+inf` background), the raw `[-1,1]` world-space normal, linear
albedo, the backward / screen-UV / Y-down motion convention (KB Decision 25),
and the categorical entity-ID map (KB Decision 9). M2 only adds **storage,
manifest, and sequence metadata** on top.

---

## 0. Pinned conventions

| Item | Value |
|---|---|
| Schema version | `"m2.0"` (session-level **and** frame-level) |
| Resolution | `512 × 288` (`[W, H]`), from the M1 exporter |
| Tick rate | `60` (KB M0 fixed physics tick) |
| Buffer storage | one **compressed `.npz`** per frame (a ZIP of `.npy` members) |
| `.npz` members | `depth`, `normal`, `albedo`, `motion`, `entity_id` |
| dtype — depth / normal / albedo / motion | `float16` (`<f2`); **depth keeps `+inf`** (half-float infinity) |
| dtype — entity_id | `int32` (`<i4`) |
| Native RGB | `float`-free 8-bit sRGB `.png` (`frame_<id>_rgb.png`) |
| Dataset root | a repo-gitignored dir: `data_pipeline/recorder/` (`.gitignore` → `data_pipeline/recorder/*`). Override with env `STRUCTRENDER_DATASET_ROOT`; falls back to `user://datasets/` |
| Session id namespace | `ds_<unix_ts>_<8 hex>` — deliberately **distinct** from the `GameLog` debug log's `session_<unix_ts>` (KB Decision 22) |
| Targets | none at M2 — every frame is `style_id: "native"`, `prompt: ""`, `target_rgb_path: null`, session `has_targets: false` (styled targets are **M3**) |

### float16 precision note

The M1 structural exporter renders **depth / normal / albedo** into a
**half-float** GPU target (`frame_structure.md` / KB Decision 26), so storing
those three channels as `float16` in the dataset loses no information the
exporter ever had. `entity_id` is an exact categorical integer and is stored as
`int32`.

`motion` is the exception: it is a CPU-side analytic `float32` computation
(`frame_structure.md` §4, KB Decision 8), so `float16` storage is a deliberate
precision reduction — bounded and harmless for training (sub-pixel ULP at the
typical ≪ 1 screen-UV magnitudes; only the large-magnitude disocclusion pixels
at reappearance boundaries degrade, and M6's confidence mask discards exactly
those). See KB Decision 27. A stage needing full-precision motion everywhere
bumps the dtype (§7).

If a future game needs `> 65504 m` depth, `frame_structure.md` §2.1
already flags the packed-float depth pass as the upgrade path — the dataset dtype
would bump with it (see §7 bump policy).

---

## 1. On-disk layout

```
<dataset_root>/
  <session_id>/                         e.g. ds_1788429583_1900d158/
    manifest.json                       session header (§3)
    manifest.jsonl                      one TrainingFrame record per line (§2), frame_id order
    events.jsonl                        entity lifecycle events (EntityEvent, frame_structure.md §5)
    frames/
      frame_00000.npz                   depth/normal/albedo/motion (f16) + entity_id (i32)
      frame_00000_rgb.png               native (unstyled) viewport RGB
      frame_00000_camera.json           { frame_id, session_id, resolution, motion_valid,
                                          view_matrix, projection_matrix, near, far }
      frame_00001.npz
      frame_00001_rgb.png
      frame_00001_camera.json
      ...
    VALIDATED                           written by the VALIDATOR only (never the recorder)
```

- `frame_id` is 0-based, monotonic, **strictly consecutive within a session**
  (no gaps — the recorder increments it by exactly 1 per recorded frame). A
  reduced recording rate shows up as a jump in `tick` / `timestamp_s`, **never**
  as a gap in `frame_id` (§5).
- The recorder **refuses to write** into a `<session_id>/` that already contains
  `manifest.jsonl` or a non-empty `frames/`, and refuses if a `VALIDATED` marker
  is present. It never overwrites and never repairs (KB §4 Dataset Recorder
  boundary).

---

## 2. `TrainingFrame` — one JSON object per line in `manifest.jsonl`

Mirrors KB §5 *Training Frame*. Fields marked **(+)** are the only additions
beyond §5; each is load-bearing data the validator needs and is documented here
rather than left implicit (KB error-handling philosophy: explicit over silent).

```jsonc
{
  "schema_version": "m2.0",
  "session_id":     "ds_1788429583_1900d158",
  "frame_id":       0,
  "timestamp_s":    0.0166667,          // true sim time = tick / tick_rate
  "tick":           1,                  // (+) true sim tick — recording-rate variation is explicit, unseen_seconds derivable
  "motion_valid":   false,              // (+) M1 whole-frame flag (frame_structure.md §4): false = no previous frame

  "structure": {
    // Each path is "<npz file>#<member>". The validator splits on '#':
    //   np.load("<session>/frames/frame_00000.npz")["<member>"]
    "depth_path":              "frames/frame_00000.npz#depth",
    "normal_path":             "frames/frame_00000.npz#normal",
    "albedo_or_material_path": "frames/frame_00000.npz#albedo",
    "motion_path":             "frames/frame_00000.npz#motion",
    "entity_id_map_path":      "frames/frame_00000.npz#entity_id"
  },

  "camera": {
    "view_matrix":       [ /* 16 floats, row-major, world -> view  (frame_structure.md §3) */ ],
    "projection_matrix": [ /* 16 floats, row-major, view  -> clip                          */ ]
  },

  "entities": [
    {
      "entity_key":   "ds_1788429583_1900d158:1:0",   // "<session_id>:<instance_id>:<generation>"
      "session_id":   "ds_1788429583_1900d158",       // (+) the structured triple, alongside entity_key
      "instance_id":  1,                              // (+)  (KB Decision 9: magnitude is meaningless / categorical)
      "generation":   0,                              // (+)
      "engine_class": "CharacterBody3D",
      "visible":      true,                           // from the entity-ID map: pixel_count > 0
      "spawned":      false,                          // a spawn event fired for this entity on THIS frame_id
      "destroyed":    false                           // a destroy event fired for this entity on THIS frame_id
    }
  ],

  "style":            { "style_id": "native", "prompt": "" },   // M2: always native, no prompt
  "native_rgb_path":  "frames/frame_00000_rgb.png",
  "target_rgb_path":  null                                       // M2: always null (styled targets are M3)
}
```

### `entity_key` is a formatting convenience, not the M7 service

`entity_key` is literally `"%s:%s:%s" % [session_id, instance_id, generation]`.
It is **not** the EntityKey Service (lookup / EPAM addressing / collision
handling) — that is **Milestone 7**. The structured `{session_id, instance_id,
generation}` triple is stored alongside so a consumer never has to parse the
string. `instance_id` / `generation` come from the M1 `EntityLifecycleAdapter`
(`instance_id` is the adapter's own monotonic counter, **not**
`Node.get_instance_id()`; `generation` bumps on engine-index reuse — a recycled
index never inherits a destroyed entity's identity).

---

## 3. Session header — `manifest.json`

```jsonc
{
  "session_id":     "ds_1788429583_1900d158",
  "schema_version": "m2.0",
  "resolution":     [512, 288],
  "frame_count":    1245,
  "tick_rate":      60,
  "has_targets":    false,
  "style_ids":      ["native"],

  "sequences": [ /* §4 */ ],

  "provenance": {
    "recorder_version":       "m2-recorder-1",
    "godot_version":          "4.7.2-stable (official)",
    "git_commit":             "5928f9a",                 // env STRUCTRENDER_GIT_COMMIT, else "unknown"
    "kb_decisions_in_force":  ["9", "22", "23", "24", "25", "26", "27", "28", "29", "30"],
    "created_unix":           1788429583
  }
}
```

---

## 4. `sequences[]` metadata

Every recorded frame belongs to exactly one recording run; `sequences[]`
describes the semantically meaningful spans within the session.

### temporal

```jsonc
{
  "sequence_id":   "temporal_0",
  "sequence_kind": "temporal",
  "frame_range":   [0, 159]        // inclusive frame_id range; DENSE and contiguous (no frame dropping within a window)
}
```

### reappearance

```jsonc
{
  "sequence_id":            "reap_camera_away_2s",
  "sequence_kind":          "reappearance",
  "frame_range":            [412, 501],
  "target_entity_key":      "ds_1788429583_1900d158:1:0",
  "target_instance_id":     1,
  "mechanism":              "camera_away",        // "camera_away" | "entity_away"
  "last_seen_frame":        423,                  // last frame_id in-range with target visible BEFORE the gap
  "first_seen_again_frame": 468,                  // first frame_id in-range with target visible AFTER the gap
  "unseen_frames":          120,                  // tick[first_seen_again] - tick[last_seen]  (sim ticks; rate-independent, exact from data)
  "unseen_seconds":         2.0,                  // timestamp_s[first_seen_again] - timestamp_s[last_seen]  (exact from data)
  "gap_ticks_configured":   120,                  // the ladder rung's configured out-of-view span (cross-check only)
  "identity_stable":        true,                 // generation_before == generation_after (instance_id is the key, so unchanged by construction)
  "generation_before":      0,
  "generation_after":       0
}
```

- `mechanism`:
  - `camera_away` — the camera turns/moves so the target leaves the view
    frustum; the target does not move.
  - `entity_away` — the target moves out of the view frustum (off-screen /
    behind an occluder); the camera does not move.
- `unseen_seconds` / `unseen_frames` are **derived from the recorded
  `timestamp_s` / `tick` of the two boundary frames**, never asserted — so they
  stay exact even where the during-gap recording rate was reduced (§5).
- The reappearance ladder sweeps **{0.5, 1, 2, 4, 8} s** (`gap_ticks_configured`
  ∈ {30, 60, 120, 240, 480} at 60 tps) for both mechanisms. (KB §11 Scenario 4's
  headline eval ladder is 1/5/10/30/60 s — that is the **eval-harness
  subagent's** dataset, not this seed fixture.)

---

## 5. Recording rate & the seed-fixture reduced-rate expedient

Every frame carries its true `timestamp_s` and `tick`, so any variation in the
recording rate is explicit in the data and `unseen_seconds` is always derivable.

- **Temporal windows** are recorded at **full rate** (1 frame per sim tick), dense
  and contiguous — no frame dropping within a window (KB §4: the recorder "does
  not sample away all temporal continuity").
- **Reappearance gaps ≤ 2 s** are recorded at **full rate**.
- **Reappearance gaps of 4 s and 8 s** MAY be recorded at a **reduced rate**
  (currently 1 frame per 8 ticks) **during the out-of-view span only**. This is a
  **seed-fixture-only expedient** to keep the seed dataset small; the
  eval-dataset during-gap policy is deferred to the eval-harness subagent. The
  `last_seen_frame` and `first_seen_again_frame` boundary frames are always
  recorded exactly, so `unseen_frames` / `unseen_seconds` remain exact.
- `frame_id` stays strictly consecutive regardless — a reduced rate is visible
  only as a jump in `tick` / `timestamp_s`.

---

## 6. Two M1 carry-over notes (contract)

1. **Partial channel dropout is the validator's job, not the recorder's.** The
   M1 exporter's own guard (`frame_structure.md` §6) aborts a frame on *total*
   channel failure (a whole pass empty / all-zero) — and the recorder propagates
   that abort loudly (it does **not** substitute a zero buffer). Detecting
   *partial* dropout — a subset of real geometry pixels reading as background
   `+inf` / id `0` while the rest of the frame is fine — is the **validator's**
   responsibility (KB M2 "Notes carried from M1").
2. **Debug log and dataset manifest are separate artifacts.** The `GameLog`
   debug log (`user://logs/session_<ts>.jsonl`, KB Decision 22) and the dataset
   manifest have **different roots**, **different `session` namespaces**
   (`session_` vs `ds_`), and **share zero files**. The recorder never writes
   under `user://logs/`, and `GameLog` never writes under the dataset root.

---

## 7. `schema_version` bump policy

`schema_version` appears at both the session (`manifest.json`) and frame
(`manifest.jsonl` line) level and must match within a session.

- **Patch is not tracked** — the string is `m<major>.<minor>` only.
- Bump the **minor** (`m2.0` → `m2.1`) for a backward-compatible addition: a new
  optional frame field, a new `sequence_kind`, a new optional provenance key.
  Old readers ignore the new field; the validator tolerates `>= m2.0`.
- Bump the **major** (`m2.x` → `m3.0`) for a breaking change: a channel dtype or
  shape change (e.g. packed-float depth), a removed/renamed field, a changed
  path scheme, a changed matrix convention, or the arrival of real targets
  (`has_targets: true`, non-null `target_rgb_path`). A validated dataset is
  immutable (KB §5) — a major bump means a **new dataset**, not an in-place
  migration.
- The bump is recorded as a dated entry in the KB Decision Log.
