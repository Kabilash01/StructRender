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
		# Catches any stray occlusion that MOVES a gap boundary (one inside the
		# pre-margin ends the visible run early; one inside the gap ends it late).
		# It cannot see a second hidden run *after* reappearance — that is what
		# `_check_one_hidden_run` below covers, once manifest.jsonl exists.
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

	# Occlusion guard (KB Decision 38): needs the whole per-frame visibility
	# timeline, so it runs here rather than inside the rung loop.
	for seq in rung_seqs:
		_check_one_hidden_run(lines, seq)

	# ---- every structure path resolves + declared shape/dtype ----
	var f0: Dictionary = JSON.parse_string(lines[0])
	_check_member(sdir, f0.structure.depth_path, [HH, W, 1], "<f2")
	_check_member(sdir, f0.structure.normal_path, [HH, W, 3], "<f2")
	_check_member(sdir, f0.structure.albedo_or_material_path, [HH, W, 3], "<f2")
	_check_member(sdir, f0.structure.motion_path, [HH, W, 2], "<f2")
	_check_member(sdir, f0.structure.entity_id_map_path, [HH, W], "<i4")
	_check_props_are_not_entities(sdir, f0, 1 if sc.e1 == null else 2)
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
## Exactly ONE hidden run per rung — the occlusion guard KB Decision 38 asked for.
##
## Decision 38 requires the placement rule it guards ("nothing on the
## camera->target sightline across the whole drift envelope") be asserted rather
## than eyeballed, and said a stray full occlusion would be caught because it
## "creates a second hidden->visible transition and the M2 validator rejects the
## session". **That was wrong** (Decision 44). `_check_sequences` in the M2
## validator locates the FIRST hidden stretch and checks only that one is clean
## (`reap.dirty_gap`); a second hidden run after reappearance is not an error
## there. Nor did the earlier assertion here cover it — it compared the derived
## gap against the configured gap, which is the same comparison already made in
## the rung loop, and both only fire when an occlusion MOVES a gap boundary. One
## sitting entirely inside the POST margin moves nothing.
##
## So this is the check that closes it: walk the target's visibility across the
## rung's own frame_range and require exactly one visible->hidden transition.
##
## NOTE ON UNITS, because the obvious check is wrong and this is why the earlier
## attempt reached for the configured gap at all: `unseen_frames` is a **tick**
## delta (Decision 29 derives it as `tick[first_seen_again] - tick[last_seen]`),
## NOT a frame_id delta. Whenever the gap is recorded at a reduced rate — always,
## in this fixture, which sets `full_rate_max = 0` — frame_ids advance once per
## recorded frame while ticks advance by the stride, so the two differ by that
## factor (measured here: 30 ticks across 4 frames). Comparing them is a unit
## error, not a fault in the data. This function counts transitions and so is
## unit-free.
func _check_one_hidden_run(lines: PackedStringArray, seq: Dictionary) -> void:
	var iid: int = int(seq.target_instance_id)
	var lo: int = int(seq.frame_range[0])
	var hi: int = int(seq.frame_range[1])
	var runs := 0
	var was_visible := true
	var found := 0
	var first_hidden := -1
	for ln in lines:
		var fr: Dictionary = JSON.parse_string(ln)
		var fid: int = int(fr.frame_id)
		if fid < lo or fid > hi:
			continue
		var vis := false
		var present := false
		for ent in fr.entities:
			if int(ent.instance_id) == iid:
				vis = bool(ent.visible)
				present = true
				break
		if present:
			found += 1
		if was_visible and not vis:
			runs += 1
			if first_hidden < 0:
				first_hidden = fid
		was_visible = vis

	# Without this the check passes vacuously: if the target were never found in
	# entities[] (a renamed field, a wrong iid), `vis` would be false for every
	# frame, giving exactly one transition out of the initial state — runs == 1,
	# green, and measuring nothing.
	H.check(found == hi - lo + 1,
		"%s: target iid %d present in entities[] for all %d frames (found %d)"
			% [seq.sequence_id, iid, hi - lo + 1, found])
	H.check(first_hidden == int(seq.last_seen_frame) + 1,
		"%s: derived first hidden frame %d == last_seen_frame + 1 (%d)"
			% [seq.sequence_id, first_hidden, int(seq.last_seen_frame) + 1])
	H.check(runs == 1,
		"%s: exactly one hidden run in frames %d-%d (got %d) — a second one means a prop "
			% [seq.sequence_id, lo, hi, runs]
		+ "fully occluded the camera->target sightline (KB Decision 38)")


## The static set added at KB Decision 38 must stay NON-ENTITY (`entity_id == 0`).
##
## That property is what keeps 13 props out of the lifecycle adapter, out of
## EPAM, and out of M7's `K_max` ranking (Decision 21) — and it is otherwise only
## implicit in `add_prop` not calling `add_to_group(ENTITY_GROUP)`. One future
## edit adding a prop via `add_entity` would silently put set dressing into the
## entity budget. Decision 38 made the occlusion rule asserted rather than
## eyeballed; this closes the same hole for the id-0 rule.
##
## Checked two ways: the id map must contain no id beyond those the manifest
## declares as entities, and the declared entity count must match the scene's
## real entity count — which the caller derives from the fixture (1 here, since
## the CI scene builds a single entity), never the entity+prop count.
func _check_props_are_not_entities(sdir: String, frame: Dictionary, expected_entities: int) -> void:
	var declared := {}
	for ent in frame.entities:
		declared[int(ent.instance_id)] = true
	H.check(declared.size() == expected_entities,
		"only the scene's %d real entities are declared, not the 13 static props (got %d)"
			% [expected_entities, declared.size()])

	var parts := String(frame.structure.entity_id_map_path).split("#", false)
	var zr := ZIPReader.new()
	if zr.open("%s/%s" % [sdir, parts[0]]) != OK:
		H.check(false, "entity_id npz opens"); return
	var raw := zr.read_file("%s.npy" % parts[1])
	zr.close()
	var hlen := raw.decode_u16(8)
	# to_int32_array() in one call: decoding 147,456 values with a per-element
	# decode_s32() loop is slow enough in GDScript to blow the scene's timeout.
	var vals := raw.slice(10 + hlen).to_int32_array()
	var seen := {}
	for v in vals:
		if v != 0:
			seen[v] = true
	var stray: Array = []
	for id in seen:
		if not declared.has(id):
			stray.append(id)
	H.check(stray.is_empty(),
		"entity_id map holds only declared entity ids — static props stay id 0 (stray: %s)" % str(stray))


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
