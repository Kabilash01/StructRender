# Teacher Target Contract — Milestone 3

Authoritative schema for a **sidecar teacher-target set**: styled RGB frames
generated offline by the diffusion teacher and consumed as supervision by M4+.
Mirrors KB §3C *Teacher Target Generation*, §4 *Offline Teacher Target
Generator*, §5 *Persistence Layer*, §6 *Replace the Offline Teacher* /
*Add a New Visual Style*, §12 *Teacher Structural Distortion*, and Decisions 6,
12, 27, 30 and **31**.

`data_pipeline/schemas/dataset_manifest.md` remains the contract for the
**structural** (m2.0) session. This file only adds the target sidecar on top of
it. The structural session is **immutable**: nothing described here ever writes
into it, re-records it, or duplicates it per style.

---

## 0. Pinned conventions

| Item | Value |
|---|---|
| Schema version | `"t1.0"` (session-level **and** frame-level) |
| Resolution | inherited from the structural session (`512 × 288`) |
| Target storage | 8-bit sRGB `.png`, one per frame, `frame_<id>_target.png` |
| Target root | `data_pipeline/teacher_targets/` (gitignored; env `STRUCTRENDER_TARGET_ROOT` overrides) |
| Directory key | `<target_root>/<style_id>/<session_id>/` — **style-keyed from day one** (KB Decision 31) |
| Style registry | `configs/teacher/<style_id>.json` (env `STRUCTRENDER_TEACHER_CONFIG_DIR` overrides) |
| Join to structure | `session_id` + `frame_id`. No copy of the structural data lives here |
| Cache key | `sha256({session_id, frame_id, style_id, teacher_version, seed})[:24]` |
| Backends | `sd15_controlnet_depth` (real), `deterministic` (CI/tests only) |
| Success marker | `ALIGNED`, written by the alignment validator only |

### Why a sidecar and not `has_targets: true` on the structural session

Decision 27's bump policy calls "the arrival of real targets" a **major** schema
bump — i.e. a new dataset. Decision 31 interprets that: targets are a *derived*
artifact over shared structural data, so making them a major bump of the
structural session would force re-recording or duplicating the one artifact
whose bit-exact frame alignment is what makes the targets valid at all, once per
style. Instead:

- `has_targets` / `target_rgb_path` in the **structural** manifest stay exactly
  as recorded at M2 (`false` / `null`);
- a consumer joins structure → target through **this** manifest;
- Decision 27's major-bump rule continues to apply to the structural dataset
  only.

---

## 1. On-disk layout

```
<target_root>/
  <style_id>/                            e.g. oil_impasto/
    <session_id>/                        e.g. ds_1788430292_4c77250c/
      manifest.json                      target-set header (§3)
      manifest.jsonl                     one TargetFrame per line (§2), frame_id order
      frames/
        frame_00000_target.png           styled RGB, same (W,H) as the native frame
        frame_00001_target.png
        ...
      ALIGNED                            written by the ALIGNMENT VALIDATOR only
```

- `frames/` may contain **only** `frame_NNNNN_target.png`. The file name *is*
  the frame-id alignment contract; a renamed target is a rejected target.
- Two styles over the same session are two sibling `<style_id>/` trees. The
  structural session is not copied into either.
- `manifest.jsonl` is written append-and-flush during generation (so an
  interrupted run resumes) and rewritten in `frame_id` order when the run ends.

---

## 2. `TargetFrame` — one JSON object per line in `manifest.jsonl`

```jsonc
{
  "schema_version":   "t1.0",
  "session_id":       "ds_1788430292_4c77250c",   // the structural session, verbatim
  "frame_id":         0,                          // must exist in that session
  "style_id":         "oil_impasto",
  "target_rgb_path":  "frames/frame_00000_target.png",
  "native_rgb_path":  "frames/frame_00000_rgb.png", // relative to the STRUCTURAL session
  "cache_key":        "9f0f4a…",                  // §4
  "content_key":      "1b7c22…",                  // §4 — hash of (native RGB, depth, teacher_version, seed)
  "content_source_frame": 0,                      // the frame whose render this PNG holds;
                                                  // == frame_id unless it was reused from a twin
  "teacher_version":  "sd15-controlnet-depth-1/225dfe3577fa",
  "seed":             20260905,
  "generated_utc":    "2026-09-05T12:00:00+00:00"
}
```

`content_key` / `content_source_frame` record content-addressed reuse (KB
Decision 33). When a session repeats a rendered moment, the twin frames share one
render and every reusing line points at the frame that produced it. On a session
with no duplicate inputs — which a correctly recorded one should be — every line
has `content_source_frame == frame_id` and nothing is reused.

`teacher_version` must be identical on every line of a set: a half-regenerated,
mixed-teacher target set is not a valid supervision source and is rejected.

---

## 3. Target-set header — `manifest.json`

```jsonc
{
  "schema_version": "t1.0",
  "style_id":       "oil_impasto",
  "style_ids":      ["oil_impasto"],          // mirrors the structural header's style_ids field
  "session_id":     "ds_1788430292_4c77250c",
  "frame_count":    990,
  "resolution":     [512, 288],
  "complete":       true,                 // false for a --limit/--stride calibration subset

  "structural_session": {
    "session_id":     "ds_1788430292_4c77250c",
    "schema_version": "m2.0",
    "frame_count":    990,
    "validated":      true,               // the VALIDATED marker was present at generation time
    "relative_hint":  "data_pipeline/recorder/ds_1788430292_4c77250c"
  },

  "teacher": {                            // KB §6 step 3: teacher metadata is VERSIONED
    "teacher_version":     "sd15-controlnet-depth-1/225dfe3577fa",
    "backend":             "sd15_controlnet_depth",
    "seed":                20260905,
    "style_config":        "oil_impasto.json",
    "style_config_digest": "225dfe…",     // sha256 of the config body (excluding "notes")
    "prompt":              "…",
    "negative_prompt":     "…",
    "reference_image":     null,
    "params":              { /* the style config's generation params */ },
    "model":               { "model_id": "…", "revision": "<commit sha>",
                             "controlnet_model_id": "…", "controlnet_revision": "<commit sha>" },
    "metadata":            { /* what the backend reported: device, dtype, licences, … */ }
  },

  "provenance": {
    "generator_version":     "m3-teacher-1",
    "git_commit":            "f078c78",
    "kb_decisions_in_force":  ["6", "9", "12", "27", "30", "31"],
    "created_unix":          1788430292,
    "updated_unix":          1788430999
  }
}
```

---

## 4. Cache semantics

KB §3C.5: *"Teacher outputs are cached because regeneration is expensive."*

```
cache_key = sha256({session_id, frame_id, style_id, teacher_version, seed})[:24]
```

A **hit** is: the manifest line for that `frame_id` records that exact key *and*
its PNG exists on disk. Hits are skipped, so a second run over an unchanged
session generates **zero** frames, and an interrupted run resumes.

`teacher_version` is `<backend_version>/<style_config_digest[:12]>`. Folding the
config digest in is deliberate: the key tuple is fixed by Decision 31, so an
edited prompt or generation parameter must invalidate the cache *through*
`teacher_version` — otherwise the old pixels would be served for a new style
definition.

There is **no eviction policy, no LRU and no cleanup pass.** A cached target is
replaced only by an explicit `--force` run. (Same discipline as the M2 validator:
tools here do not garbage-collect the user's data.)

Diffusion generation is seed-pinned and reproducible on a fixed GPU/driver but
**not bit-exact across machines** (Decision 31) — which is exactly why targets
are cached and version-keyed rather than regenerated on demand. The
`deterministic` backend *is* bit-exact, which is what makes the test suite
reproducible.

---

## 5. Alignment validation

`python -m data_pipeline.teacher validate <target_dir>`

**Report-only.** It never repairs a target, never regenerates one, never mutates
the target set, and never touches the structural session. The single file it may
write is `ALIGNED`, and only on a clean run over a *complete* target set (never
under `--limit`). This mirrors the M2 dataset validator exactly (KB §4, error
philosophy "explicit over silent, no auto-repair").

### Everything is compared in the edge/gradient domain

A teacher target is *supposed* to look different from the native frame, so
pixel-domain SSIM/LPIPS against native RGB would penalise the stylisation we
asked for. Only **structure** must survive, so all comparison is between the
target's edge map and edge maps derived exactly from the engine's own structural
buffers.

Two structural maps, and they are not the same thing:

| Map | Content | Used for |
|---|---|---|
| **reference** `R` | depth-discontinuity edges ∪ `entity_id` boundaries | recall, per-entity silhouette support |
| **support** `S` | `R` ∪ normal creases ∪ albedo/material boundaries | precision, spurious-edge ratio |

`S` is a superset on purpose: a target edge that lands on a material boundary or
a surface crease is *explained by the scene* even though it is not a silhouette.
Scoring every painterly stroke along a texture boundary as "hallucinated
geometry" would make the §12 guard fire on legitimate stylisation. Both maps come
only from the M2 structural buffers — never from the native RGB and never from
the target. `entity_id` is used only through **inequality** (is this a different
entity?), never as a magnitude (KB Decision 9).

Target edges: BT.601 luma → Sobel (normalised by the kernel weight, so `0.008`
means "≈2/255 intensity step per pixel") → Canny-style non-maximum suppression,
then two thresholds: `edge_threshold` (the permissive set) and
`strong_edge_threshold` (the confident set).

The two halves of the F-measure use **different** target edge sets, and this is
calibrated, not arbitrary (KB Decision 32):

- **recall** — reference edges matched by the *permissive* set. "Did the layout
  survive at all." A legitimate boundary in a low-contrast scene can be rendered
  softly, so requiring a strong edge here would fail honest targets.
- **precision** — fraction of *strong* target edges that have structural
  support. "Are the confident edges explained by the scene." This is
  `1 − spurious_edge_ratio` by construction.

Precision on the permissive set is computed and reported as `edge_precision_all`
but is **not** gated: on the measured seed data it scores how much low-amplitude
brushwork a painterly teacher added — stylisation, not structural distortion —
and gating it rejects the real teacher while accepting a 3 px misaligned target.

### The four gates

| Gate | Quantity | Rule |
|---|---|---|
| 1 | boundary F-measure at 2 px (recall vs `R`, precision vs `S`) | `>= boundary_f` |
| 2 | spurious-edge ratio — fraction of **strong** target edges with no structural support within 2 px | `<= spurious_ratio` |
| 3 | per-entity silhouette support, for every entity with `>= 200` visible px | `>= silhouette_support` |
| 4 | **stylisation distance** — colour-histogram L1 between target and *native* frame | `>= min_stylisation` |

**Gate 4 is not optional, and a replacement validator that omits it is wrong.**
Gates 1–3 only ask *did the structure survive*, and the perfect way to satisfy
all three is to change nothing: measured on the seed session, the **untouched
native frame** scores `boundary_f = 0.976` with `spurious = 0.000` — better than
any real target. Without a floor on how far the target moved from the native
frame, "turn the teacher down until it passes" is the optimum and the M3 exit
criterion is met by a teacher that does nothing. The histogram term carries the
gate because the scene is dark: a real restyle shifts the palette a long way
while moving mean `|Δpixel|` only ~2 levels out of 255 (the pixel term is
reported as `stylisation_pix` for diagnostics only). See **KB Decision 32,
Finding 1**, and `test_passthrough_native_frame_is_rejected`, which asserts that
gate 4 — not a structural gate — is what rejects a passthrough.

Thresholds and their calibration are pinned in **KB Decision 32**. Gate 1 is
undefined (`NaN`, never a failure) on a frame with no visible geometry; a target
that draws *nothing* scores `F = 0` and fails, rather than being excused as
undefined. Gate 3 is likewise skipped on a frame carrying no entity of at least
`min_entity_px` — on the seed session that is 295 of 990 frames, so roughly 30 %
of frames are ungated on entity silhouette.

**Two measurement limits a consumer must know about:**

- **Detection floor on translation.** Measured floors: **2 px diagonal, 5 px
  x-only** (KB Decision 41 — improved from 3 px / 8 px on the retired scene,
  once the enriched scene added vertical structure). Shifts *at or above* those
  values are rejected; a systematic misalignment **below** 5 px horizontally
  still passes all four gates and would reach training as supervision. The
  x-only floor is the weaker one because a long horizontal ground/sky boundary
  is near-invariant under horizontal translation. Both floors are asserted by
  `test_alignment_rejects.py` (`DIAGONAL_FLOOR_PX` / `X_ONLY_FLOOR_PX`), so a
  regression is caught rather than merely documented.
- **Partial circularity.** The teacher is conditioned on the exported depth
  buffer, and gate 1's recall is measured against a reference map derived from
  that same buffer. The `support` map adds normal and albedo discontinuities,
  which are *not* conditioning inputs, so this is partly mitigated — but the
  metric cannot detect a structural distortion that ControlNet-depth is itself
  blind to.

By default **one** frame failing **one** gate rejects the whole set
(`--max-fail-frac 0.0`). The tolerance is an explicit operator knob, in the same
spirit as the M2 validator's `--dropout-tolerance`.

### The native-RGB sanity anchor

The engine's own RGB is by construction perfectly aligned with its own
structural buffers, so running the metrics on it must score near the ceiling. If
it does not, the reference edge map — not the teacher — is what is wrong. This
is asserted as a test, not just documented.

---

## 6. Boundaries (KB §4)

- The teacher **never runs inside the game loop** and is **never a runtime
  dependency**. It is offline-only; TensorRT deployment does not know it exists.
- The teacher **never owns structure or entity identity**. It consumes recorded
  structural buffers read-only and returns pixels.
- Godot remains the sole source of gameplay truth. Nothing in this pipeline
  feeds back into the game.
- Entity IDs stay **addresses** (masks / keys), never numeric input channels.
- The alignment validator **reports only**.

---

## 7. `schema_version` bump policy

`schema_version` is `t<major>.<minor>` at both header and frame level and must
match within a set.

- Bump the **minor** for a backward-compatible addition (a new optional field, a
  new provenance key). The validator tolerates a newer minor.
- Bump the **major** for a breaking change: a changed path scheme, a removed or
  renamed field, a different target file format, or a change to the meaning of
  the cache key.
- A **new style** is *not* a bump — that is the whole point of Decision 31's
  style-keyed layout. Adding `configs/teacher/<new>.json` and running the
  generator is the entire procedure (KB §6 *Add a New Visual Style* steps 1-3);
  it is covered by `test_second_style_by_config_alone`.
- A **new teacher** is *not* a bump either — it is a new `teacher_version`,
  which invalidates the cache for the affected targets (KB §6 *Replace the
  Offline Teacher* step 4: "regenerate only required targets").
- Each bump gets a dated entry in the KB Decision Log.
