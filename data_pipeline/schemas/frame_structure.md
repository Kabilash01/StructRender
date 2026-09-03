# Structural Export Contract — Milestone 1

Authoritative schema for the per-frame structural buffers produced by the Godot
structural exporter (`game_project/instrumentation/structural_exporter.gd`) and
the entity-lifecycle events produced by
`game_project/instrumentation/entity_lifecycle_adapter.gd`.

Mirrors KB §4 *Structural Export Module* / *Entity Lifecycle Adapter* and KB §5
*Training Frame*. This file is the single source of truth for M1 exported
*values*, ranges, and conventions. Anything not listed here is **not** exported at
M1 (roughness/metallic, native RGB, dataset manifest — later milestones).

Config: `configs/data/structural_export.json` (resolution, sentinel, id base).

---

## 0. Pinned conventions (do not change without a KB Decision Log entry)

| Item | Value |
|---|---|
| Resolution | **512 × 288** (`[W, H]`), config-driven |
| Render path | Forward+ (KB Decision 23) |
| Engine | Godot `4.7.2-stable` (KB §12) |
| Buffer memory layout | row-major, `index = (y * W + x) * C + c`, origin **top-left**, **Y-down** |
| NPY dtype (continuous) | `float32` (`<f4`), little-endian |
| NPY dtype (entity id) | `int32` (`<i4`), little-endian |
| Colour space (albedo) | **linear** (not sRGB) |
| Normal storage | **world-space, unit-length, raw `[-1, 1]`** (not packed `[0,1]`) |
| Depth | **linear eye-space Z in metres**, raw, background = `+inf` |

---

## 1. `FrameStructure` (in-memory dict, per frame)

Produced by `StructuralExporter.export_frame(frame_id)`. Shapes mirror KB §4.

```python
FrameStructure = {
    "schema_version":      "m1.0",
    "session_id":          str,             # e.g. "session_1717430000" (deterministic override allowed for tests)
    "frame_id":            int,             # monotonic, starts at 0
    "resolution":          [W, H],          # [512, 288]

    "depth":               float32[H, W, 1],   # linear eye-space Z, metres, +inf = no geometry
    "normal":              float32[H, W, 3],   # world-space unit vector, raw [-1,1]; (0,0,0) where depth==+inf
    "albedo_or_material":  float32[H, W, 3],   # linear base colour; (0,0,0) where depth==+inf
    "motion":              float32[H, W, 2],   # backward, screen-UV [0,1], Y-down (see §4)
    "motion_valid":        bool,               # false on frame 0 / no previous frame (documented absence, NOT a failure)
    "entity_id_map":       int32[H, W],        # per-pixel stable instance_id; 0 = background

    "camera_view":         float32[4, 4],      # row-major; world -> view      (see §3)
    "camera_projection":   float32[4, 4],      # row-major; view  -> clip       (see §3)
    "camera_near":         float,              # metres
    "camera_far":          float,              # metres

    "entities": [                                # one entry per *currently registered* entity
        {
            "instance_id":  int,                # stable address, >= 1 (KB Decision 9: magnitude is meaningless)
            "generation":   int,                # >= 0, incremented on engine-index reuse
            "engine_class": str,                # Godot class name, e.g. "CharacterBody3D"
            "pixel_count":  int,                # pixels this entity owns in entity_id_map this frame
            "visible":      bool,               # pixel_count > 0
        },
        ...
    ],
}
```

### Debug dumps (synthetic scenes only, M1)

`StructuralExporter.dump_frame(fs, dir)` writes, for eyeballing S1–S3:

```
<dir>/frame_<id>_depth.npy              float32 [H, W, 1]
<dir>/frame_<id>_normal.npy             float32 [H, W, 3]
<dir>/frame_<id>_albedo.npy             float32 [H, W, 3]
<dir>/frame_<id>_motion.npy             float32 [H, W, 2]
<dir>/frame_<id>_entity_id.npy          int32   [H, W]
<dir>/frame_<id>_camera.json            {view, projection, near, far, frame_id, session_id, motion_valid}
<dir>/frame_<id>_*.png                  viewable previews (see scripts/visualize_frame.py)
```

No manifest, no JSONL frame store, no Parquet — that is **Milestone 2**.

---

## 2. Channel definitions

### 2.1 depth — `float32 [H, W, 1]`

- **Value:** linear eye-space Z, i.e. `-(view_matrix * world_point).z`, in engine
  units (metres). Positive in front of the camera. This is the *planar* distance
  along the camera view axis (**not** Euclidean ray length) so it round-trips
  exactly through `Camera3D.project_position(screen, depth)`.
- **No normalization** is applied here. Store raw metres.
- **Background / no geometry:** `+inf` (numpy `float('inf')`). **Never `0`.**
  The exporter renders geometry against a pure-black environment and treats any
  pixel with rendered eye-Z `<= 0.0` as background.
- **Range limit:** the GPU pass stores eye-Z in a half-float render target, so
  the maximum finite representable depth is ≈ 65504 m; beyond that it saturates
  to `+inf` (indistinguishable from background). All M1 synthetic scenes and the
  Arena scene are < 100 m deep, well inside this limit. Flagged as accepted M1
  debt; a packed-float depth pass can replace this later if a game needs it.
- `camera_near` / `camera_far` are stored per frame in metadata.

### 2.2 normal — `float32 [H, W, 3]`

- **Value:** world-space surface normal, **unit length**, stored **raw in
  `[-1, 1]`** (NOT packed to `[0,1]`).
- Computed as `normalize(MODEL_NORMAL_MATRIX * NORMAL)` in the override shader,
  written directly to a half-float target (negatives preserved).
- **Background:** `(0, 0, 0)` (not unit — callers must mask on `depth == +inf`).
- Non-uniform mesh scale is not corrected beyond `MODEL_NORMAL_MATRIX`; M1
  scenes use uniform scale.

### 2.3 albedo_or_material — `float32 [H, W, 3]`

- **Value:** linear base colour (RGB). Roughness / metallic are **explicitly
  deferred** — this channel is albedo only at M1.
- Per mesh, base colour is resolved as:
  1. `MeshInstance3D.get_meta("structrender_albedo")` (a `Color`, interpreted
     **linear**) if present, else
  2. the active `StandardMaterial3D.albedo_color` converted `srgb_to_linear()`,
     else
  3. white `(1, 1, 1)`.
- Written unshaded to a half-float target, so the stored value is the linear
  colour with no tonemap / no sRGB OETF.
- **Background:** `(0, 0, 0)`.

### 2.4 entity_id_map — `int32 [H, W]`

- **Value:** the owning entity's stable `instance_id` (see §5), or `0` for
  background and for non-entity geometry (arena floor, walls, spawner).
- **Categorical (KB Decision 9):** magnitude has no meaning, no ordering, never
  fed to a network as a continuous channel. Its only M1 job is masks +
  visibility lifecycle events.
- `instance_id` starts at **1** and is monotonic per session.
- GPU encoding: `id` is split little-endian across the RGB channels of an
  unshaded pass (`r = id & 255`, `g = (id>>8) & 255`, `b = (id>>16) & 255`),
  rendered to a half-float target (each channel 0–255, exact), decoded by
  `round()` on readback. Supports `instance_id` up to 2^24 − 1.
- **Validation:** every non-background pixel must decode to a *registered*
  `instance_id`; otherwise the exporter aborts the frame (see §6).

### 2.5 motion — see §4.

---

## 3. Camera matrices — `float32 [4, 4]`, row-major

Target multiplication order (column-vector convention):

```
clip_point = camera_projection @ camera_view @ world_point      # world_point = [x, y, z, 1]^T
ndc        = clip_point.xyz / clip_point.w
screen_px  = ( (ndc.x + 1) / 2 * W ,  (1 - ndc.y) / 2 * H )     # Y-down pixels
```

- `camera_view` = `Camera3D.global_transform.affine_inverse()`, written as a 4×4
  with bottom row `(0, 0, 0, 1)`.
  Element `camera_view[r][c]`:
  - `r < 3, c < 3`  → `basis.get_column(c)[r]` of the inverse transform
  - `r < 3, c == 3` → `origin[r]` of the inverse transform
  - `r == 3`        → `(0, 0, 0, 1)`
- `camera_projection` = `Camera3D.get_camera_projection()` (a Godot `Projection`,
  which already maps view → clip for `Projection * Vector4`).
  Element `camera_projection[r][c]` = `projection[c][r]` (Godot `Projection` is
  column-major: `projection[c]` is column `c` as a `Vector4`).
  - Godot clip space: `x, y ∈ [-w, w]`, `z ∈ [0, w]` (Vulkan convention). The
    screen mapping above only uses `x, y`; consumers needing linear depth should
    use the `depth` channel, not `ndc.z`.
- Verified in synthetic scene **S1**: manual `projection @ view @ world_pt`
  screen projection matches `Camera3D.unproject_position(world_pt)` within 1 px.

---

## 4. motion — `float32 [H, W, 2]` (KB Decision 8, KB §12 top fragile area)

**Convention (pinned — KB Decision 25):**

- **Backward:** `motion[p]` points from pixel `p` in frame `t` to where that
  same surface point was in frame `t-1`.
- **Screen-UV, normalized `[0, 1]`**, **Y-down** (same axis as the buffers).
- Includes **camera motion + rigid per-object motion**.
- Computed **analytically from engine transforms** — never pixel-estimated
  (KB Decision 8 / Decision 7).

**Exact definition** (as implemented):

```
for pixel p = (x, y) in frame t:
    d          = depth[p]                                  # linear eye-space Z, metres
    if d == +inf:  motion[p] = (0, 0);  continue           # background/sky: undefined parallax -> 0 (pinned, see below)

    eid        = entity_id_map[p]                          # 0 => background => rigid transform = identity
    if eid != 0 and entity eid has no frame t-1 transform:  motion[p] = (0, 0);  continue
                                                           # entity spawned / became registered this frame:
                                                           # no valid previous correspondence -> 0 (documented, like frame 0)

    world_p    = Camera3D.project_position((x + 0.5, y + 0.5), d)      # current camera unprojection

    M_t        = global_transform of entity eid at frame t     (identity if eid == 0)
    M_{t-1}    = global_transform of entity eid at frame t-1    (identity if eid == 0)
    world_p0   = (M_{t-1} @ M_t^{-1}) @ world_p            # rigid object motion; identity for static bg

    clip_prev  = P_{t-1} @ V_{t-1} @ [world_p0, 1]         # previous-frame camera (stored matrices)
    ndc_prev   = clip_prev.xyz / clip_prev.w
    uv_prev    = ( (ndc_prev.x + 1) / 2 , (1 - ndc_prev.y) / 2 )        # Y-down UV
    uv_curr    = ( (x + 0.5) / W , (y + 0.5) / H )

    motion[p]  = uv_prev - uv_curr                         # BACKWARD, UV, Y-down
```

### Frame 0 / no correspondence — pinned (KB Decision 25)

Whenever a pixel has **no valid previous-frame correspondence**, `motion[p]` is a
documented `(0, 0)` — never a guess, never reprojected from partial data. Three
cases:

- **Frame 0** (no `t-1` frame at all): `motion` is an all-zero buffer **and**
  `motion_valid = false`. `motion_valid` stays a **whole-frame** flag ("no
  previous frame at all"); there is no per-pixel validity channel.
- **Background / sky** (`depth == +inf`): `motion[p] = (0, 0)`. The surface is
  not reprojected at the far plane for camera-only motion. **Rationale:**
  zero-history background is masked by the M6 disocclusion confidence mask
  anyway, so synthesising camera-only background motion buys nothing while adding
  a far-plane-parallax failure mode. This is a pinned choice, not an
  implementation shortcut.
- **Entity with no `t-1` transform** (spawned or first registered this frame, so
  `_prev_entity_xforms` has no entry for its `instance_id`): `motion[p] = (0, 0)`,
  same treatment as frame 0. Using `M_{t-1} = identity` while `M_t` is real would
  reinterpret object-local coords as world coords and emit a meaningless non-zero
  vector. The entity recovers correct analytic motion on the **next** frame, once
  it has a `t-1` transform. `motion_valid` is unaffected (it remains a
  whole-frame flag); the frame is still emitted normally.

### Verification

- Sign check (S2): camera moves `+X` in world between `t-1` and `t`; static
  geometry then appears to move `-X` on screen from `t-1` to `t`, so the backward
  vector `uv_prev - uv_curr` points `+X` (`motion.x > 0`). Asserted in S2.
- Y check (S2/S3): camera/object moving up in world → surface moves toward
  smaller pixel-y (up) between `t-1` and `t` → `uv_prev` has larger `v` →
  `motion.y > 0` (Y-down UV). Asserted in S3.
- Warp check (S4): `warp(frame_{t-1}, motion) ≈ frame_t` within photometric
  tolerance (checkerboard test).
- New-entity check (S7): an entity spawned at frame 1 has all its pixels at
  `motion == (0, 0)` exactly (no `t-1` transform), an existing entity's motion is
  unperturbed, sky (`+inf`) pixels are `(0, 0)`; moving that entity at frame 2
  recovers the correct analytic motion.

---

## 5. Entity identity & `EntityEvent`

Produced by `EntityLifecycleAdapter`. **Not** the EntityKey string service
(`<session_id>:<instance_id>:<generation>` formatting is M7) — M1 only emits the
structured event dict below.

```python
EntityEvent = {
    "session_id":  str,
    "instance_id": int,     # stable, monotonic per session, starts at 1  (NOT Godot get_instance_id())
    "generation":  int,     # starts at 0
    "event":       str,     # one of: "spawn" | "destroy" | "visible" | "hidden"
    "frame_id":    int,
}
```

Written one-per-line to `events.jsonl` (`EntityLifecycleAdapter.flush_events(path)`).

### Rules

- **Registration:** an entity is any node in the Godot group
  `structrender_entity`. In the Arena scene that is `Player`, `Coin_0..4`,
  `Enemy_0`. `Arena`, `Spawner`, `MainCamera`, `WorldEnvironment`,
  `DirectionalLight3D` are **not** entities.
- `instance_id` is assigned by the adapter's own monotonic counter. It is **not**
  `Node.get_instance_id()` (which is a recyclable low-level engine index).
- `generation` starts at 0. When the adapter observes a **previously-freed engine
  index reused by a different live node**, that new node gets a **fresh
  `instance_id`** and `generation = previous_record.generation + 1` (KB §12
  "Entity ID Reuse", KB EntityKey Service boundary: a recycled index never
  inherits the destroyed entity's identity).
- **`spawn`** / **`destroy`** track existence (node enters / leaves the group or
  is freed).
- **`visible`** / **`hidden`** track on-screen presence and are **derived from
  the entity-ID map** (`pixel_count` crossing `0`). They are **fully decoupled**
  from `spawn`/`destroy`: an entity leaving the camera view emits `hidden`, never
  `destroy`; re-entering emits `visible`, never `spawn`, and its `instance_id` /
  `generation` are unchanged.
- **Test seam:** `EntityLifecycleAdapter.engine_index_provider : Callable` (takes
  a `Node`, returns an int) defaults to `Node.get_instance_id`. Tests override it
  to force a deterministic engine-index collision (S5b).

---

## 6. No silent channel failure (KB error-handling philosophy)

`StructuralExporter` validates every frame before returning it. On **any** of the
following it raises (`push_error` + `assert`) and returns `null` — **no frame is
emitted, no zero/stale buffer is substituted**:

- a pass produced no image, or an all-zero image where geometry is expected;
- wrong shape or dtype for any channel;
- non-finite values in `depth` (other than the `+inf` sentinel), `normal`,
  `albedo`, or the camera matrices;
- `normal` not unit-length (‖n‖ ∉ [0.9, 1.1]) on a non-background pixel;
- `entity_id_map` contains a non-background value that is not a registered
  `instance_id`;
- an entity node in `structrender_entity` has no adapter record (unregistered).

The diagnostic names the failing channel. Verified by synthetic scene **S6**.

---

## 7. KB Decision Log entry (Decision 25 — added to docs/StructRender_KB.md §9)

> ### Decision 25: Motion Vector Convention Pinned (2026-09-03)
>
> - **Context:** KB §12 flags "Motion Coordinate Convention" as the top fragile
>   area (previous/current direction, NDC vs UV, Y-axis inversion, camera-only vs
>   object motion). Milestone 1 computes motion vectors; leaving the convention
>   implicit invites warp-direction bugs in Milestone 6 (temporal reprojection),
>   where the sign of the field silently determines whether history is pulled
>   from the right place.
>
> - **Chosen Path:** Motion is **backward** (frame `t` → frame `t-1`),
>   **screen-UV normalized `[0,1]`**, **Y-down** (same axis as all structural
>   buffers), includes **camera + rigid-object** motion, and is computed
>   **analytically from engine transforms** (unproject current depth → apply
>   `M_{t-1} M_t^{-1}` → reproject with the previous-frame camera), never
>   estimated from pixels (consistent with Decision 7/8). Frame 0 emits a
>   zero buffer with `motion_valid = false` (documented absence, not a failure).
>   Full definition in `data_pipeline/schemas/frame_structure.md` §4.
>
> - **Rejected Alternatives:** forward motion (t → t+1, needs future state);
>   NDC-space or pixel-space units (resolution-coupled, and NDC Y is up while the
>   buffers are Y-down — a guaranteed sign bug); camera-only motion (breaks on
>   moving entities); pixel-estimated flow (redundant, non-deterministic,
>   violates Decision 7).
>
> - **Verification:** synthetic scenes S2 (known camera translation, sign +
>   direction), S3 (known object translation, moving vs static pixels),
>   S4 (checkerboard backward-warp photometric alignment).
