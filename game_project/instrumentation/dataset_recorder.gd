extends Node
## Milestone 2 — Dataset Recorder (recorder side only; the validator is a
## separate Python task). KB §3B, §4 "Dataset Recorder", §5 "Training Frame".
##
## Drives the M1 StructuralExporter as an ISOLATED capture mode (KB Decision 26:
## each export_frame advances the SceneTree ~8x and swaps every material, so the
## sim must be stepped under the recorder/scenario's control, never in a live
## play loop). Not an autoload (KB Decision 24) — a scenario script under
## game_project/tests/dataset/ adds this node and calls it explicitly.
##
## Per recorded frame it writes:
##   frames/frame_<id>.npz          depth/normal/albedo/motion (f16) + entity_id (i32)
##   frames/frame_<id>_rgb.png      native (unstyled) viewport RGB
##   frames/frame_<id>_camera.json  view / projection / near / far
##   one TrainingFrame line -> manifest.jsonl
## On finish(): manifest.json (session header) + events.jsonl (from the adapter).
##
## Boundaries (KB §4): does NOT validate/repair data, does NOT train, does NOT
## generate teacher targets (every frame style is "native", target_rgb_path null),
## does NOT overwrite an existing session or a VALIDATED one. A failed exporter
## frame aborts loudly — no zero-fill.
##
## Loaded via preload (no class_name), matching the M1 instrumentation.

const Npz := preload("res://instrumentation/npz_writer.gd")

const SCHEMA_VERSION := "m2.0"
const RECORDER_VERSION := "m2-recorder-1"
const KB_DECISIONS_IN_FORCE := ["9", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

var session_id: String = ""
var dataset_root: String = ""          # absolute dir; <root>/<session_id>/ is the session dir
var resolution: Vector2i = Vector2i(512, 288)
var schema_version: String = SCHEMA_VERSION    # m3.0 for the 768x432 dataset (KB Decision 100)
var tick_rate: int = 60
var git_commit: String = "unknown"

var _exporter: Node
var _adapter: Node
var _src_cam: Camera3D

var _native_vp: SubViewport
var _native_cam: Camera3D

var _session_dir: String = ""
var _frames_dir: String = ""
var _manifest: FileAccess
var _frame_count: int = 0
var _started: bool = false

# per-frame log, for deriving reappearance stats from the data (never asserted)
var _frame_log: Array[Dictionary] = []          # {frame_id, tick, timestamp_s, vis_by_iid, gen_by_iid}
var _entity_meta: Dictionary = {}               # iid -> {generation, engine_class}
var _sequences: Array[Dictionary] = []


func configure(p_session_id: String, p_root: String, exporter: Node, adapter: Node,
		src_cam: Camera3D, res: Vector2i = Vector2i(512, 288), p_tick_rate: int = 60,
		p_schema_version: String = SCHEMA_VERSION) -> void:
	session_id = p_session_id
	dataset_root = p_root
	_exporter = exporter
	_adapter = adapter
	_src_cam = src_cam
	resolution = res
	tick_rate = p_tick_rate
	schema_version = p_schema_version
	var env_commit := OS.get_environment("STRUCTRENDER_GIT_COMMIT")
	if env_commit != "":
		git_commit = env_commit


## Create the session dir. Refuses (returns false, writes nothing) if the session
## already has frames / a manifest, or if a VALIDATED marker is present. Returns
## true on a clean start.
func begin() -> bool:
	assert(session_id != "" and dataset_root != "" and _exporter != null,
		"DatasetRecorder.configure() not called")
	_session_dir = "%s/%s" % [dataset_root.rstrip("/"), session_id]
	_frames_dir = "%s/frames" % _session_dir

	if FileAccess.file_exists("%s/VALIDATED" % _session_dir):
		push_error("DatasetRecorder: %s has a VALIDATED marker — immutable, refusing to write." % _session_dir)
		return false
	if FileAccess.file_exists("%s/manifest.jsonl" % _session_dir):
		push_error("DatasetRecorder: %s already has a manifest.jsonl — refusing to overwrite." % _session_dir)
		return false
	if DirAccess.dir_exists_absolute(_frames_dir):
		var existing := DirAccess.get_files_at(_frames_dir)
		if existing.size() > 0:
			push_error("DatasetRecorder: %s already contains %d files — refusing to overwrite." % [_frames_dir, existing.size()])
			return false

	var err := DirAccess.make_dir_recursive_absolute(_frames_dir)
	if err != OK and err != ERR_ALREADY_EXISTS:
		push_error("DatasetRecorder: cannot create %s (err %d)" % [_frames_dir, err])
		assert(false, "DatasetRecorder: cannot create session dir")
		return false

	_manifest = FileAccess.open("%s/manifest.jsonl" % _session_dir, FileAccess.WRITE)
	if _manifest == null:
		push_error("DatasetRecorder: cannot open manifest.jsonl (err %d)" % FileAccess.get_open_error())
		assert(false, "DatasetRecorder: cannot open manifest")
		return false

	_setup_native_viewport()
	_started = true
	return true


## Record one frame at simulation tick `tick`. The scenario advances the
## deterministic sim (moves entities / camera) BEFORE calling this. Returns the
## TrainingFrame dict that was written. Aborts loudly on exporter failure.
func record_frame(tick: int) -> Dictionary:
	assert(_started, "DatasetRecorder.begin() not called / returned false")
	var frame_id := _frame_count
	var timestamp_s := float(tick) / float(tick_rate)

	var stem := "frame_%05d" % frame_id

	# Native RGB first, while every mesh still has its real material — the
	# exporter below swaps material_override on all meshes for its passes.
	await _capture_native_rgb("%s/%s_rgb.png" % [_frames_dir, stem])

	var fs: Dictionary = await _exporter.export_frame(frame_id)
	if fs.is_empty():
		push_error("DatasetRecorder: exporter aborted frame %d (%s) — nothing recorded, no zero-fill." % [frame_id, _exporter.last_error])
		assert(false, "DatasetRecorder: exporter frame failed")
		return {}

	var w := resolution.x
	var h := resolution.y

	Npz.write_npz("%s/%s.npz" % [_frames_dir, stem], Npz.frame_members(fs, h, w))
	_write_camera_json("%s/%s_camera.json" % [_frames_dir, stem], fs)

	var events_this_frame := _events_at(frame_id)
	var entities := _build_entities(fs, events_this_frame)

	var rel := "frames/%s.npz" % stem
	var record := {
		"schema_version": schema_version,
		"session_id": session_id,
		"frame_id": frame_id,
		"timestamp_s": timestamp_s,
		"tick": tick,
		"motion_valid": bool(fs.motion_valid),
		"structure": {
			"depth_path": "%s#depth" % rel,
			"normal_path": "%s#normal" % rel,
			"albedo_or_material_path": "%s#albedo" % rel,
			"motion_path": "%s#motion" % rel,
			"entity_id_map_path": "%s#entity_id" % rel,
		},
		"camera": {
			"view_matrix": Array(fs.camera_view),
			"projection_matrix": Array(fs.camera_projection),
		},
		"entities": entities,
		"style": {"style_id": "native", "prompt": ""},
		"native_rgb_path": "frames/%s_rgb.png" % stem,
		"target_rgb_path": null,
	}
	_manifest.store_line(JSON.stringify(record))
	_manifest.flush()

	_log_frame(frame_id, tick, timestamp_s, fs)
	_frame_count += 1
	return record


func add_temporal_sequence(sequence_id: String, first_frame: int, last_frame: int) -> Dictionary:
	var seq := {
		"sequence_id": sequence_id,
		"sequence_kind": "temporal",
		"frame_range": [first_frame, last_frame],
	}
	_sequences.append(seq)
	return seq


## Derive a reappearance sequence's semantics from the recorded frame log — never
## from what the scenario "meant to do". `gap_ticks_configured` is the ladder
## rung (0.5/1/2/4/8 s * tick_rate) kept only for cross-checking.
func add_reappearance_sequence(sequence_id: String, target_iid: int, mechanism: String,
		gap_ticks_configured: int, first_frame: int, last_frame: int) -> Dictionary:
	var rows: Array[Dictionary] = []
	for r in _frame_log:
		if r.frame_id >= first_frame and r.frame_id <= last_frame:
			rows.append(r)
	rows.sort_custom(func(a, b): return a.frame_id < b.frame_id)

	var last_seen := -1
	var first_again := -1
	var seen_hidden := false
	for r in rows:
		var vis: bool = bool(r.vis_by_iid.get(target_iid, false))
		if not seen_hidden:
			if vis:
				last_seen = r.frame_id
			else:
				seen_hidden = true
		elif vis:
			first_again = r.frame_id
			break

	var a := _row_for_frame(rows, last_seen)
	var b := _row_for_frame(rows, first_again)
	var unseen_frames := -1
	var unseen_seconds := -1.0
	var identity_stable := false
	var gen_before := -1
	var gen_after := -1
	if not a.is_empty() and not b.is_empty():
		unseen_frames = int(b.tick) - int(a.tick)
		unseen_seconds = float(b.timestamp_s) - float(a.timestamp_s)
		gen_before = int(a.gen_by_iid.get(target_iid, -1))
		gen_after = int(b.gen_by_iid.get(target_iid, -1))
		identity_stable = gen_before >= 0 and gen_before == gen_after

	var meta: Dictionary = _entity_meta.get(target_iid, {})
	var seq := {
		"sequence_id": sequence_id,
		"sequence_kind": "reappearance",
		"frame_range": [first_frame, last_frame],
		"target_entity_key": "%s:%d:%d" % [session_id, target_iid, int(meta.get("generation", 0))],
		"target_instance_id": target_iid,
		"mechanism": mechanism,
		"last_seen_frame": last_seen,
		"first_seen_again_frame": first_again,
		"unseen_frames": unseen_frames,
		"unseen_seconds": unseen_seconds,
		"gap_ticks_configured": gap_ticks_configured,
		"identity_stable": identity_stable,
		"generation_before": gen_before,
		"generation_after": gen_after,
	}
	_sequences.append(seq)
	return seq


func finish() -> void:
	assert(_started, "DatasetRecorder.begin() not called")
	if _manifest != null:
		_manifest.flush()
		_manifest.close()
		_manifest = null
	_adapter.flush_events("%s/events.jsonl" % _session_dir)

	var header := {
		"session_id": session_id,
		"schema_version": schema_version,
		"resolution": [resolution.x, resolution.y],
		"frame_count": _frame_count,
		"tick_rate": tick_rate,
		"has_targets": false,
		"style_ids": ["native"],
		"sequences": _sequences,
		"provenance": {
			"recorder_version": RECORDER_VERSION,
			"godot_version": Engine.get_version_info().get("string", "unknown"),
			"git_commit": git_commit,
			"kb_decisions_in_force": KB_DECISIONS_IN_FORCE,
			"created_unix": int(Time.get_unix_time_from_system()),
		},
	}
	var f := FileAccess.open("%s/manifest.json" % _session_dir, FileAccess.WRITE)
	if f == null:
		push_error("DatasetRecorder: cannot write manifest.json (err %d)" % FileAccess.get_open_error())
		assert(false, "DatasetRecorder: cannot write manifest header")
		return
	f.store_string(JSON.stringify(header, "  "))
	f.close()
	_started = false


func session_dir() -> String:
	return _session_dir


func frame_count() -> int:
	return _frame_count


# ---------------------------------------------------------------------------

func _setup_native_viewport() -> void:
	_native_vp = SubViewport.new()
	_native_vp.size = resolution
	_native_vp.msaa_3d = Viewport.MSAA_DISABLED
	_native_vp.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	_native_vp.use_taa = false
	_native_vp.use_hdr_2d = false  # RGBA8 -> sRGB-encoded 8-bit, correct for a native RGB .png
	_native_vp.positional_shadow_atlas_size = 0
	_native_vp.world_3d = _src_cam.get_world_3d()
	_native_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(_native_vp)
	_native_cam = Camera3D.new()
	_native_vp.add_child(_native_cam)


func _capture_native_rgb(path: String) -> void:
	_native_cam.global_transform = _src_cam.global_transform
	_native_cam.projection = _src_cam.projection
	_native_cam.fov = _src_cam.fov
	_native_cam.size = _src_cam.size
	_native_cam.near = _src_cam.near
	_native_cam.far = _src_cam.far
	_native_cam.keep_aspect = _src_cam.keep_aspect
	_native_cam.current = true
	await get_tree().process_frame
	await get_tree().process_frame
	await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var img := _native_vp.get_texture().get_image()
	var err := img.save_png(path)
	if err != OK:
		push_error("DatasetRecorder: native RGB save_png failed (%d) for %s" % [err, path])
		assert(false, "DatasetRecorder: native RGB capture failed")


func _write_camera_json(path: String, fs: Dictionary) -> void:
	var cam := {
		"frame_id": fs.frame_id,
		"session_id": session_id,
		"resolution": [resolution.x, resolution.y],
		"motion_valid": fs.motion_valid,
		"view_matrix": Array(fs.camera_view),
		"projection_matrix": Array(fs.camera_projection),
		"near": fs.camera_near,
		"far": fs.camera_far,
	}
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(cam, "  "))
		f.close()


func _events_at(frame_id: int) -> Dictionary:
	var out := {"spawn": {}, "destroy": {}}
	for e in _adapter.events():
		if int(e.frame_id) != frame_id:
			continue
		if e.event == "spawn":
			out.spawn[int(e.instance_id)] = true
		elif e.event == "destroy":
			out.destroy[int(e.instance_id)] = true
	return out


func _build_entities(fs: Dictionary, events_this_frame: Dictionary) -> Array:
	var out: Array = []
	var alive_iids := {}
	for ent in fs.entities:
		var iid := int(ent.instance_id)
		var gen := int(ent.generation)
		alive_iids[iid] = true
		_entity_meta[iid] = {"generation": gen, "engine_class": String(ent.engine_class)}
		out.append({
			"entity_key": "%s:%d:%d" % [session_id, iid, gen],
			"session_id": session_id,
			"instance_id": iid,
			"generation": gen,
			"engine_class": String(ent.engine_class),
			"visible": bool(ent.visible),
			"spawned": events_this_frame.spawn.has(iid),
			"destroyed": false,
		})
	for iid in events_this_frame.destroy:
		if alive_iids.has(iid):
			continue
		var meta: Dictionary = _entity_meta.get(iid, {"generation": 0, "engine_class": "Node"})
		out.append({
			"entity_key": "%s:%d:%d" % [session_id, iid, int(meta.generation)],
			"session_id": session_id,
			"instance_id": iid,
			"generation": int(meta.generation),
			"engine_class": String(meta.engine_class),
			"visible": false,
			"spawned": false,
			"destroyed": true,
		})
	return out


func _log_frame(frame_id: int, tick: int, timestamp_s: float, fs: Dictionary) -> void:
	var vis_by_iid := {}
	var gen_by_iid := {}
	for ent in fs.entities:
		vis_by_iid[int(ent.instance_id)] = bool(ent.visible)
		gen_by_iid[int(ent.instance_id)] = int(ent.generation)
	_frame_log.append({
		"frame_id": frame_id,
		"tick": tick,
		"timestamp_s": timestamp_s,
		"vis_by_iid": vis_by_iid,
		"gen_by_iid": gen_by_iid,
	})


func _row_for_frame(rows: Array, frame_id: int) -> Dictionary:
	if frame_id < 0:
		return {}
	for r in rows:
		if r.frame_id == frame_id:
			return r
	return {}
