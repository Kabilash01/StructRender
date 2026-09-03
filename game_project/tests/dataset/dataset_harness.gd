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
		exporter: Node, adapter: Node, src_cam: Camera3D) -> Node:
	var rec := Recorder.new()
	tree.root.add_child(rec)
	rec.configure(session_id, root, exporter, adapter, src_cam,
		Vector2i(W_RES, H_RES), TICK_RATE)
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


## Build the standard reappearance/temporal scene: floor + backdrop + up to two
## entities + a camera looking down -Z. Returns {world, cam, e0, e1}.
static func build_scene(tree: SceneTree, two_entities: bool) -> Dictionary:
	var world := H.make_world(tree)
	await tree.process_frame
	add_light_env(world)
	H.add_prop(world, Vector3(0, -1, -8), Vector3(40, 0.5, 40), Color(0.25, 0.25, 0.25))
	H.add_prop(world, Vector3(0, 4, -16), Vector3(60, 20, 0.5), Color(0.20, 0.22, 0.30))
	var e0 := H.add_entity(world, "Target_0", Vector3(-1.5, 0.0, -6.0), 1.4, Color(0.90, 0.65, 0.20))
	var e1: Node3D = null
	if two_entities:
		e1 = H.add_entity(world, "Target_1", Vector3(2.0, 0.0, -7.0), 1.2, Color(0.25, 0.55, 0.85))
	var cam := H.make_camera(world, Vector3(0, 1.0, 0.5), Vector3(0, 0.0, -6.0))
	return {"world": world, "cam": cam, "e0": e0, "e1": e1}


## Record `count` consecutive frames with the sim advancing 1 tick/frame and all
## entities left in place. `S` carries {rec, tick}. Returns the last frame_id.
static func record_still(S: Dictionary, count: int) -> int:
	var last := -1
	for _i in count:
		S.tick += 1
		var r: Dictionary = await S.rec.record_frame(S.tick)
		last = int(r.frame_id)
	return last


## Drive one reappearance rung and register the sequence.
##   S : {rec, tick, cam, cam_home, target, target_home, occluder_pos}
## `mechanism` is "camera_away" or "entity_away". Returns the sequence dict.
static func run_rung(S: Dictionary, sequence_id: String, gap_ticks: int, mechanism: String,
		pre_frames: int, post_frames: int) -> Dictionary:
	# reset to the visible home state
	S.cam.global_transform = S.cam_home
	S.target.global_position = S.target_home
	var f0: int = S.rec.frame_count()

	# --- PRE: target fully visible ---
	await record_still(S, pre_frames)

	# --- GAP: target out of view for exactly `gap_ticks` sim ticks ---
	var full_rate_max: int = int(S.get("full_rate_max", FULL_RATE_MAX_TICKS))
	var reduced := gap_ticks > full_rate_max
	for k in range(1, gap_ticks + 1):
		S.tick += 1
		if k < gap_ticks:
			if mechanism == "camera_away":
				S.cam.global_transform = _camera_aimed_away(S.cam_home)
			else:
				S.target.global_position = S.occluder_pos
		else:
			# last gap tick: back to the visible home state -> first_seen_again
			S.cam.global_transform = S.cam_home
			S.target.global_position = S.target_home
		if k == gap_ticks or not reduced or k % REDUCED_STRIDE == 0:
			await S.rec.record_frame(S.tick)

	# --- POST: target visible again ---
	await record_still(S, post_frames - 1)
	var f1: int = S.rec.frame_count() - 1

	var target_iid: int = S.rec_adapter.instance_id_for(S.target)
	return S.rec.add_reappearance_sequence(sequence_id, target_iid, mechanism, gap_ticks, f0, f1)


static func _camera_aimed_away(home: Transform3D) -> Transform3D:
	# same eye position, yawed 165 deg about world-up: the target ends up behind.
	var t := home
	t.basis = Basis(Vector3.UP, deg_to_rad(165.0)) * home.basis
	return t
