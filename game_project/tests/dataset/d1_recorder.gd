extends SceneTree
## Milestone 2 — dataset recorder verification (GPU: exporter + native-RGB
## viewport need a real rendering device, like the S1-S7 scenes).
##
## Records a tiny fixture (1 short temporal window + the {0.5,1,2}s camera_away
## rungs for one entity) and asserts:
##   - deterministic: same scenario twice -> byte-identical .npz + .png, and
##     manifest.jsonl identical modulo the session_id,
##   - refuse-to-overwrite + VALIDATED-marker immutability,
##   - manifest well-formed: schema_version at both levels, every *_path resolves
##     and each npz member has the declared shape + dtype,
##   - temporal window: strictly consecutive frame_id,
##   - reappearance rungs: target visible true->false->true, instance_id /
##     generation constant across the gap, unseen_seconds == span / tick_rate,
##   - debug-log vs dataset-manifest separation.
##
## Run by scripts/test_dataset.sh (env STRUCTRENDER_DATASET_ROOT points at a
## temp dir). Loaded with preload deps, no class_name (M0/M1 pattern).

const H := preload("res://tests/structural/structural_harness.gd")
const DH := preload("res://tests/dataset/dataset_harness.gd")

const W := DH.W_RES
const HH := DH.H_RES
const RATE := DH.TICK_RATE


func _initialize() -> void:
	await _test_determinism_and_immutability()
	await _test_fixture_and_manifest()
	H.finish(self)


# -------------------------------------------------------------------------
func _record_still_session(sid: String, root: String, frames: int) -> Node:
	var sc: Dictionary = await DH.build_scene(self, false)
	await process_frame
	var adapter := H.make_adapter(self, sid)
	var ex := H.make_exporter(self, sc.cam, sc.world, adapter, sid)
	var rec := DH.make_recorder(self, sid, root, ex, adapter, sc.cam)
	H.check(rec.begin(), "begin() clean start for %s" % sid)
	var S := {"rec": rec, "tick": 0}
	for _i in frames:
		S.tick += 1
		await rec.record_frame(S.tick)
	rec.add_temporal_sequence("t_still", 0, frames - 1)
	rec.finish()
	sc.world.queue_free()
	adapter.queue_free()
	ex.queue_free()
	rec.queue_free()
	await process_frame
	return rec


func _test_determinism_and_immutability() -> void:
	var root := DH.dataset_root()
	var sid_a := DH.new_session_id()
	await _record_still_session(sid_a, root, 4)
	var dir_a := "%s/%s" % [root, sid_a]

	# --- refuse to overwrite an existing session ---
	var sc2: Dictionary = await DH.build_scene(self, false)
	await process_frame
	var ad2 := H.make_adapter(self, sid_a)
	var ex2 := H.make_exporter(self, sc2.cam, sc2.world, ad2, sid_a)
	var rec2 := DH.make_recorder(self, sid_a, root, ex2, ad2, sc2.cam)
	var npz0_before := FileAccess.get_file_as_bytes("%s/frames/frame_00000.npz" % dir_a)
	H.check(not rec2.begin(), "begin() refuses a session that already has a manifest")
	var npz0_after := FileAccess.get_file_as_bytes("%s/frames/frame_00000.npz" % dir_a)
	H.check(npz0_before == npz0_after and npz0_before.size() > 0,
		"refused begin() clobbered nothing")
	sc2.world.queue_free(); ad2.queue_free(); ex2.queue_free(); rec2.queue_free()
	await process_frame

	# --- VALIDATED marker blocks recording ---
	var sid_v := DH.new_session_id()
	var vdir := "%s/%s" % [root, sid_v]
	DirAccess.make_dir_recursive_absolute(vdir)
	FileAccess.open("%s/VALIDATED" % vdir, FileAccess.WRITE).store_string("{}")
	var sc3: Dictionary = await DH.build_scene(self, false)
	await process_frame
	var ad3 := H.make_adapter(self, sid_v)
	var ex3 := H.make_exporter(self, sc3.cam, sc3.world, ad3, sid_v)
	var rec3 := DH.make_recorder(self, sid_v, root, ex3, ad3, sc3.cam)
	H.check(not rec3.begin(), "begin() refuses a session with a VALIDATED marker")
	sc3.world.queue_free(); ad3.queue_free(); ex3.queue_free(); rec3.queue_free()
	await process_frame

	# --- determinism: a second identical run is byte-identical modulo session_id ---
	var sid_b := DH.new_session_id()
	await _record_still_session(sid_b, root, 4)
	var dir_b := "%s/%s" % [root, sid_b]
	# The buffer CONTENT is bit-exact reproducible; the .npz zip container embeds a
	# creation time, so compare decoded members, not container bytes (see schema).
	for i in 4:
		for key in ["depth", "normal", "albedo", "motion", "entity_id"]:
			var ma := _member_bytes("%s/frames/frame_%05d.npz" % [dir_a, i], key)
			var mb := _member_bytes("%s/frames/frame_%05d.npz" % [dir_b, i], key)
			H.check(ma == mb and ma.size() > 0, "frame %d npz member '%s' bit-identical across runs" % [i, key])
		var ra := Image.new(); ra.load("%s/frames/frame_%05d_rgb.png" % [dir_a, i])
		var rb := Image.new(); rb.load("%s/frames/frame_%05d_rgb.png" % [dir_b, i])
		H.check(ra.get_data() == rb.get_data() and ra.get_data().size() > 0,
			"frame %d native rgb pixels identical across runs" % i)
	var man_a := FileAccess.get_file_as_string("%s/manifest.jsonl" % dir_a).replace(sid_a, "SID")
	var man_b := FileAccess.get_file_as_string("%s/manifest.jsonl" % dir_b).replace(sid_b, "SID")
	H.check(man_a == man_b and man_a.length() > 0,
		"manifest.jsonl frame content identical across runs (session_id normalised)")


# -------------------------------------------------------------------------
func _test_fixture_and_manifest() -> void:
	var root := DH.dataset_root()
	var sid := DH.new_session_id()
	var sc: Dictionary = await DH.build_scene(self, false)
	await process_frame
	var adapter := H.make_adapter(self, sid)
	var ex := H.make_exporter(self, sc.cam, sc.world, adapter, sid)
	var rec := DH.make_recorder(self, sid, root, ex, adapter, sc.cam)
	H.check(rec.begin(), "fixture: begin() clean")

	var S := {
		"rec": rec, "rec_adapter": adapter, "tick": 0,
		"cam": sc.cam, "cam_home": sc.cam.global_transform,
		"target": sc.e0, "target_home": sc.e0.global_position,
		"occluder_pos": Vector3(1000, 0, -6),
		"full_rate_max": 0,   # CI smoke test: reduced-rate gaps throughout (fast)
	}

	# temporal window
	var t0: int = rec.frame_count()
	await DH.record_still(S, 12)
	var t1: int = int(rec.frame_count()) - 1
	rec.add_temporal_sequence("temporal_0", t0, t1)

	# {0.5, 1, 2}s camera_away rungs on the one entity
	var rung_seqs: Array = []
	for s in [0.5, 1.0, 2.0]:
		var gap := int(round(s * RATE))
		var seq: Dictionary = await DH.run_rung(S, "reap_cam_%.1fs" % s, gap, "camera_away", 5, 5)
		rung_seqs.append(seq)
		H.check(seq.last_seen_frame >= 0 and seq.first_seen_again_frame > seq.last_seen_frame,
			"rung %.1fs: visible -> hidden -> visible in the recorded data" % s)
		H.check(seq.identity_stable and seq.generation_before == seq.generation_after,
			"rung %.1fs: instance_id / generation unchanged across the gap" % s)
		H.check(absf(seq.unseen_seconds - float(gap) / RATE) <= 1.5 / RATE,
			"rung %.1fs: unseen_seconds %.4f == %d ticks / %d (±1 frame)" % [s, seq.unseen_seconds, gap, RATE])
		H.check(absi(int(seq.unseen_frames) - gap) <= 1,
			"rung %.1fs: unseen_frames %d == configured %d (±1)" % [s, seq.unseen_frames, gap])

	rec.finish()
	var sdir: String = rec.session_dir()

	# ---- manifest.json header ----
	var hdr: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("%s/manifest.json" % sdir))
	H.check(hdr.schema_version == "m2.0", "manifest.json schema_version == m2.0")
	H.check(int(hdr.frame_count) == rec.frame_count(), "manifest.json frame_count matches")
	H.check(hdr.has_targets == false, "manifest.json has_targets == false")
	H.check(hdr.style_ids == ["native"], "manifest.json style_ids == [native]")
	H.check(int(hdr.resolution[0]) == W and int(hdr.resolution[1]) == HH,
		"manifest.json resolution == [%d, %d]" % [W, HH])
	var kinds := {}
	for q in hdr.sequences:
		kinds[q.sequence_kind] = int(kinds.get(q.sequence_kind, 0)) + 1
	H.check(kinds.get("temporal", 0) == 1 and kinds.get("reappearance", 0) == 3,
		"manifest.json sequences: 1 temporal + 3 reappearance (got %s)" % str(kinds))
	H.check(hdr.provenance.recorder_version != "" and hdr.provenance.kb_decisions_in_force.has("26"),
		"manifest.json provenance present (recorder_version + KB decisions)")

	# ---- manifest.jsonl frames ----
	var lines := FileAccess.get_file_as_string("%s/manifest.jsonl" % sdir).split("\n", false)
	H.check(lines.size() == rec.frame_count(), "manifest.jsonl has one line per frame")
	var prev_fid := -1
	for ln in lines:
		var fr: Dictionary = JSON.parse_string(ln)
		H.check(fr.schema_version == "m2.0", "frame %d schema_version present" % fr.frame_id)
		if prev_fid >= 0:
			H.check(int(fr.frame_id) == prev_fid + 1, "frame_id strictly consecutive (%d -> %d)" % [prev_fid, fr.frame_id])
		prev_fid = int(fr.frame_id)
		H.check(fr.style.style_id == "native" and fr.target_rgb_path == null,
			"frame %d style native / no target" % fr.frame_id)

	# temporal window frame_ids are a gapless run
	H.check(prev_fid == rec.frame_count() - 1, "last frame_id == frame_count - 1 (no gaps anywhere)")

	# ---- every structure path resolves + declared shape/dtype ----
	var f0: Dictionary = JSON.parse_string(lines[0])
	_check_member(sdir, f0.structure.depth_path, [HH, W, 1], "<f2")
	_check_member(sdir, f0.structure.normal_path, [HH, W, 3], "<f2")
	_check_member(sdir, f0.structure.albedo_or_material_path, [HH, W, 3], "<f2")
	_check_member(sdir, f0.structure.motion_path, [HH, W, 2], "<f2")
	_check_member(sdir, f0.structure.entity_id_map_path, [HH, W], "<i4")
	var rgb := Image.new()
	H.check(rgb.load("%s/%s" % [sdir, f0.native_rgb_path]) == OK and rgb.get_size() == Vector2i(W, HH),
		"native_rgb_path loads to %dx%d" % [W, HH])
	var non_black := false
	for _t in 200:
		if rgb.get_pixel(randi() % W, randi() % HH).r > 0.02:
			non_black = true; break
	H.check(non_black, "native rgb is a real lit render (not all black)")
	H.check(f0.camera.view_matrix.size() == 16 and f0.camera.projection_matrix.size() == 16,
		"frame camera matrices are 4x4")

	# ---- reappearance visible transition in the manifest entities ----
	var seq0: Dictionary = rung_seqs[0]
	var saw_true := false
	var saw_false_after_true := false
	var saw_true_again := false
	for ln in lines:
		var fr: Dictionary = JSON.parse_string(ln)
		if int(fr.frame_id) < int(seq0.frame_range[0]) or int(fr.frame_id) > int(seq0.frame_range[1]):
			continue
		for ent in fr.entities:
			if int(ent.instance_id) != int(seq0.target_instance_id):
				continue
			if ent.visible:
				if saw_false_after_true:
					saw_true_again = true
				else:
					saw_true = true
			elif saw_true:
				saw_false_after_true = true
	H.check(saw_true and saw_false_after_true and saw_true_again,
		"manifest entities[] show the target visible -> hidden -> visible")

	# ---- debug-log vs dataset-manifest separation ----
	var glog: Node = root_glog()
	var glog_path := "" if glog == null else String(glog.get_log_path())
	H.check(sid.match("ds_*") and not sid.begins_with("session_"),
		"dataset session_id is in the ds_ namespace, not the GameLog session_ namespace")
	H.check(glog_path == "" or not glog_path.begins_with(sdir),
		"GameLog wrote outside the dataset session dir (%s)" % glog_path)
	var logs_dir := ProjectSettings.globalize_path("user://logs")
	H.check(not sdir.begins_with(logs_dir), "dataset session dir is not under user://logs")
	var names := {}
	for fn in DirAccess.get_files_at(sdir):
		names[fn] = true
	H.check(names.has("manifest.json") and names.has("manifest.jsonl") and names.has("events.jsonl"),
		"dataset session dir has exactly the manifest/events artifacts (got %s)" % str(names.keys()))
	for fn in names:
		H.check(not fn.begins_with("session_"), "no GameLog-style file leaked into the dataset dir: %s" % fn)


# -------------------------------------------------------------------------
func root_glog() -> Node:
	return root.get_node_or_null("GameLog")


func _member_bytes(zpath: String, key: String) -> PackedByteArray:
	var zr := ZIPReader.new()
	if zr.open(zpath) != OK:
		return PackedByteArray()
	var raw := zr.read_file("%s.npy" % key)
	zr.close()
	return raw


func _check_member(sdir: String, spec: String, want_shape: Array, want_descr: String) -> void:
	var parts := spec.split("#", false)
	var zpath := "%s/%s" % [sdir, parts[0]]
	var key := parts[1] if parts.size() > 1 else ""
	var zr := ZIPReader.new()
	if zr.open(zpath) != OK:
		H.check(false, "npz opens: %s" % zpath); return
	var raw := zr.read_file("%s.npy" % key)
	zr.close()
	H.check(raw.size() > 0, "npz member '%s' present in %s" % [key, parts[0]])
	if raw.size() < 12:
		return
	var hlen := raw.decode_u16(8)
	var hdr := raw.slice(10, 10 + hlen).get_string_from_ascii()
	var descr := hdr.get_slice("'descr': '", 1).get_slice("'", 0)
	var shp_str := hdr.get_slice("'shape': (", 1).get_slice(")", 0)
	var dims: Array = []
	for p in shp_str.split(",", false):
		var v := p.strip_edges()
		if v != "":
			dims.append(int(v))
	H.check(descr == want_descr, "%s dtype %s == %s" % [key, descr, want_descr])
	H.check(dims == want_shape, "%s shape %s == %s" % [key, str(dims), str(want_shape)])
