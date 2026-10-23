extends RefCounted
## Shared helpers for the Milestone 2 dataset-recorder scenario scripts
## (game_project/tests/dataset/). Builds on structural_harness.gd (check/finish/
## make_world/add_entity/make_camera/make_exporter/make_adapter) and adds:
##   - the dataset root resolution + recorder wiring,
##   - a lit scene environment so the native RGB capture is a real render,
##   - the reappearance-ladder driver (pre / controlled gap / post).
##
## These run via `godot --rendering-driver vulkan --script ...` (NOT --headless —
## the exporter + native-RGB viewport need a real GPU). See scripts/test_dataset.sh.
##
## Loaded with preload (no class_name), same as the structural harness.

const H := preload("res://tests/structural/structural_harness.gd")
const Recorder := preload("res://instrumentation/dataset_recorder.gd")

const W_RES := 512
const H_RES := 288
const TICK_RATE := 60

## Ladder rungs (KB §11 Scenario 4 uses 1/5/10/30/60 s for the eval headline;
## the M2 seed fixture uses the shorter, cheaper {0.5,1,2,4,8} s sweep the M2
## task specifies — the eval-dataset ladder is the eval-harness subagent's job).
const RUNGS_SECONDS := [0.5, 1.0, 2.0, 4.0, 8.0]

## Gaps longer than this (in ticks) may be recorded at a reduced rate in the
## seed fixture ONLY (see dataset_manifest.md). Every recorded frame still carries
## its true tick/timestamp_s so unseen_seconds stays exact.
const FULL_RATE_MAX_TICKS := 120          # 2 s at 60 tps
const REDUCED_STRIDE := 8


## Dataset session-id namespace (KB Decision 22 debug log is `session_<ts>` —
## this is deliberately distinct). ds_<unix_ts>_<shorthash>.
static func new_session_id() -> String:
	var ts := int(Time.get_unix_time_from_system())
	var salt := str(ts) + "_" + str(Time.get_ticks_usec())
	var short := salt.sha256_text().substr(0, 8)
	return "ds_%d_%s" % [ts, short]


static func dataset_root() -> String:
	var env := OS.get_environment("STRUCTRENDER_DATASET_ROOT")
	if env != "":
		return env
	return ProjectSettings.globalize_path("user://datasets")


static func make_recorder(tree: SceneTree, session_id: String, root: String,
		exporter: Node, adapter: Node, src_cam: Camera3D,
		res: Vector2i = Vector2i(W_RES, H_RES), schema: String = Recorder.SCHEMA_VERSION) -> Node:
	var rec := Recorder.new()
	tree.root.add_child(rec)
	rec.configure(session_id, root, exporter, adapter, src_cam, res, TICK_RATE, schema)
	return rec


## A directional light + ambient environment on the scene's world, so the native
## RGB capture is a lit render (the structural exporter uses its own black env).
static func add_light_env(world: Node3D) -> void:
	var we := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.09, 0.11, 0.15, 1.0)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.6, 0.6, 0.6, 1.0)
	env.ambient_light_energy = 0.6
	we.environment = env
	world.add_child(we)
	var sun := DirectionalLight3D.new()
	world.add_child(sun)
	sun.rotation_degrees = Vector3(-45, -30, 0)
	# Load-bearing, not tidiness (KB Decision 38, "Explicitly cut"): the structural
	# passes are unlit, so a shadow edge is a strong native-RGB edge with NO
	# counterpart in `R` or `S` — gate 2 would score it as hallucinated geometry.
	# False is the engine default; set explicitly so enabling it is a visible edit.
	sun.shadow_enabled = false


## Static, non-entity set dressing (all `entity_id == 0`) — KB Decision 38.
##
## The scene was previously one flat floor plus one flat backdrop. When the
## target entity leaves the frame, that left the depth buffer as two constant
## planes: nothing for ControlNet-depth to anchor on, almost no reference edges,
## and so any prompt strong enough to restyle also invented geometry (Decision
## 37). Every piece below earns its place by adding a depth discontinuity, a
## silhouette, or an albedo boundary in exactly those sparse frames.
##
## Three placement rules, all load-bearing:
##
## 1. BOTH camera poses are covered. `_camera_aimed_away` yaws 165° about
##    world-up, so props sitting only in front of the home pose would leave the
##    `camera_away` gaps — half the observed failure cluster — as empty as
##    before. The `+Z` group exists solely for that pose.
## 2. NOTHING blocks the camera→target sightline. Visibility is
##    `pixel_count > 0`, so a prop that *fully* occludes the target for even one
##    frame inside a margin creates a second hidden→visible transition. This
##    used to add "and the M2 validator rejects the session (Decision 29's
##    single contiguous gap)" — it does not; the validator only checks the FIRST
##    hidden stretch is clean. The guard is `_check_one_hidden_run` in
##    `d1_recorder.gd` (KB Decision 44).
##    The target drifts within x ∈ [-2.1, -0.9], z ≈ -6.4..-5.6; the camera
##    drifts x ∈ [-0.35, 0.35] at z ≈ 0.5. Everything here is therefore kept
##    off the central corridor (|x| ≥ 3.2 for anything nearer than z = -12) or
##    placed behind the target plane.
## 3. DEPTH STAYS IN USABLE RANGE. M4 feeds `1/(1+d)`, so 40 m maps to 0.024
##    against a background sentinel of exactly 0 — beyond that "varied depth" is
##    numerically indistinguishable from sky. Everything sits ~4-32 m out.
static func _build_static_set(world: Node3D) -> void:
	# --- floor, broken into three sections at different heights/tilts so the
	# ground plane itself carries depth jumps AND normal creases (both R and S)
	H.add_prop(world, Vector3(0, -1.0, -8), Vector3(26, 0.5, 26), Color(0.26, 0.24, 0.22))
	H.add_prop(world, Vector3(-11, -0.55, -13), Vector3(12, 0.5, 22), Color(0.32, 0.30, 0.25),
		Vector3(0, 0, -4.0))
	H.add_prop(world, Vector3(11, -0.72, -14), Vector3(12, 0.5, 24), Color(0.20, 0.22, 0.26),
		Vector3(3.0, 0, 0))

	# --- backdrop, stepped rather than one flat wall
	H.add_prop(world, Vector3(-6, 3.5, -19), Vector3(26, 18, 0.5), Color(0.18, 0.20, 0.28))
	H.add_prop(world, Vector3(13, 5.0, -23), Vector3(26, 22, 0.5), Color(0.23, 0.21, 0.19))

	# --- mid-depth blocks flanking the corridor: silhouettes + depth steps that
	# survive when the entities are gone. Kept to |x| >= 3.2 so the target's
	# drift envelope is never fully occluded.
	H.add_prop(world, Vector3(-4.6, 0.1, -10.5), Vector3(2.2, 3.4, 2.2), Color(0.55, 0.30, 0.18))
	H.add_prop(world, Vector3(5.2, -0.2, -9.0), Vector3(1.8, 2.2, 1.8), Color(0.22, 0.42, 0.38),
		Vector3(0, 22.0, 0))
	H.add_prop(world, Vector3(-7.8, 0.9, -15.0), Vector3(3.0, 5.0, 2.4), Color(0.42, 0.44, 0.50))
	H.add_prop(world, Vector3(8.0, 0.6, -16.5), Vector3(2.6, 4.4, 2.6), Color(0.60, 0.52, 0.28),
		Vector3(0, -15.0, 0))

	# --- the +Z group: only these are in frame during a `camera_away` gap.
	# Without them that half of the failure cluster stays as empty as before.
	H.add_prop(world, Vector3(0, 4.0, 14), Vector3(30, 20, 0.5), Color(0.21, 0.19, 0.24))
	H.add_prop(world, Vector3(-5.0, 0.2, 8.0), Vector3(2.4, 3.6, 2.4), Color(0.48, 0.34, 0.26))
	H.add_prop(world, Vector3(4.4, -0.1, 6.5), Vector3(2.0, 2.6, 2.0), Color(0.30, 0.36, 0.46),
		Vector3(0, 18.0, 0))
	H.add_prop(world, Vector3(0.5, -0.8, 5.0), Vector3(14, 0.5, 16), Color(0.28, 0.26, 0.23),
		Vector3(-2.5, 0, 0))


## Build the standard reappearance/temporal scene: floor + backdrop + up to two
## entities + a camera looking down -Z. Returns {world, cam, e0, e1}.
static func build_scene(tree: SceneTree, two_entities: bool) -> Dictionary:
	var world := H.make_world(tree)
	await tree.process_frame
	add_light_env(world)
	_build_static_set(world)
	var e0 := H.add_entity(world, "Target_0", Vector3(-1.5, 0.0, -6.0), 1.4, Color(0.90, 0.65, 0.20))
	var e1: Node3D = null
	if two_entities:
		e1 = H.add_entity(world, "Target_1", Vector3(2.0, 0.0, -7.0), 1.2, Color(0.25, 0.55, 0.85))
	var cam := H.make_camera(world, Vector3(0, 1.0, 0.5), Vector3(0, 0.0, -6.0))
	return {"world": world, "cam": cam, "e0": e0, "e1": e1}


## Record `count` consecutive frames with the sim advancing 1 tick/frame.
##
## The camera and target are kept gently MOVING (see `_apply_drift`) rather than
## frozen. Holding them still made 830 of the seed session's 990 frames
## byte-identical copies of just three images — the reappearance *semantics* were
## correct but the set carried almost no visual diversity, which makes a
## downstream overfit test pass vacuously. Amplitudes are small and bounded so
## the target's visibility state is unchanged. `S` carries {rec, tick}.
## Returns the last frame_id.
static func record_still(S: Dictionary, count: int) -> int:
	var last := -1
	for _i in count:
		S.tick += 1
		_apply_drift(S, true)
		var r: Dictionary = await S.rec.record_frame(S.tick)
		last = int(r.frame_id)
	return last


## Small continuous motion around the current phase's base pose, driven by the
## sim tick so it is fully deterministic and re-recordable.
##
## `target_visible` selects the base pose: the home pair when the target is meant
## to be on screen, the aimed-away camera / off-screen target position when it is
## not. Amplitudes are deliberately far inside the frustum margin (the target
## sits ~1.5 units off-centre in a view whose horizontal half-extent is ~8.9
## units at that depth), so drift NEVER changes whether the target is visible —
## the property every reappearance sequence's semantics depend on. If that ever
## stops holding, the M2 validator recomputes last_seen/first_seen_again from the
## recorded data and rejects the session rather than silently mislabelling it.
static func _apply_drift(S: Dictionary, target_visible: bool) -> void:
	var t := float(S.tick) / float(TICK_RATE)
	var base_cam: Transform3D = S.cam_home if target_visible else _camera_aimed_away(S.cam_home)

	var yaw := deg_to_rad(6.0) * sin(t * 0.9)
	var xf := base_cam
	xf.basis = Basis(Vector3.UP, yaw) * base_cam.basis
	xf.origin = base_cam.origin + Vector3(0.35 * sin(t * 0.7), 0.18 * sin(t * 1.3), 0.0)
	S.cam.global_transform = xf

	var base_pos: Vector3 = S.target_home if target_visible else S.occluder_pos
	S.target.global_position = base_pos + Vector3(0.6 * sin(t * 1.1), 0.22 * sin(t * 1.7), 0.4 * sin(t * 0.5))

	# a second entity, when the scene has one, drifts on its own phase
	if S.has("target2") and S.target2 != null:
		var h2: Vector3 = S.target2_home
		S.target2.global_position = h2 + Vector3(0.5 * sin(t * 0.8 + 2.1), 0.2 * sin(t * 1.5 + 0.7), 0.3 * sin(t * 0.6 + 1.3))


## Drive one reappearance rung and register the sequence.
##   S : {rec, tick, cam, cam_home, target, target_home, occluder_pos}
## `mechanism` is "camera_away" or "entity_away". Returns the sequence dict.
static func run_rung(S: Dictionary, sequence_id: String, gap_ticks: int, mechanism: String,
		pre_frames: int, post_frames: int) -> Dictionary:
	# reset to the visible home state
	S.cam.global_transform = S.cam_home
	S.target.global_position = S.target_home
	var f0: int = S.rec.frame_count()

	# --- PRE: target fully visible (and drifting) ---
	await record_still(S, pre_frames)

	# --- GAP: target out of view for exactly `gap_ticks` sim ticks ---
	# The scene keeps moving through the gap too, so the frames recorded while the
	# target is hidden are not copies of each other either. Only the BASE pose
	# switches; `_apply_drift` keeps the target off screen for the whole span and
	# returns it to the visible base on the final tick.
	var full_rate_max: int = int(S.get("full_rate_max", FULL_RATE_MAX_TICKS))
	var reduced := gap_ticks > full_rate_max
	for k in range(1, gap_ticks + 1):
		S.tick += 1
		if mechanism == "camera_away":
			# Only the CAMERA moves for this mechanism (dataset_manifest.md §4):
			# the target is frozen for the whole gap, final tick included.
			var was: Vector3 = S.target.global_position
			_apply_drift(S, k >= gap_ticks)
			S.target.global_position = was
		else:
			# Only the TARGET moves for this mechanism: the camera is frozen for
			# the whole gap, final tick included.
			var cam_was: Transform3D = S.cam.global_transform
			_apply_drift(S, k >= gap_ticks)
			S.cam.global_transform = cam_was
		if k == gap_ticks or not reduced or k % REDUCED_STRIDE == 0:
			await S.rec.record_frame(S.tick)

	# --- POST: target visible again (and drifting) ---
	await record_still(S, post_frames - 1)
	var f1: int = S.rec.frame_count() - 1

	var target_iid: int = S.rec_adapter.instance_id_for(S.target)
	return S.rec.add_reappearance_sequence(sequence_id, target_iid, mechanism, gap_ticks, f0, f1)


static func _camera_aimed_away(home: Transform3D) -> Transform3D:
	# same eye position, yawed 165 deg about world-up: the target ends up behind.
	var t := home
	t.basis = Basis(Vector3.UP, deg_to_rad(165.0)) * home.basis
	return t
