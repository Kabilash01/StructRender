extends SceneTree
## Milestone 2 — produce the seed dataset (a regenerable test fixture, NOT
## committed; scripts/record_seed_dataset.sh runs this).
##
## Contents:
##   - temporal_0 : one dense ~160-frame window, camera panning + one entity
##     moving, no frame dropping (KB §4 "does not sample away all temporal
##     continuity"),
##   - reap_camera_away_{0.5,1,2,4,8}s and reap_entity_away_{...}s : the
##     reappearance ladder for one target entity, both mechanisms. Gaps <= 2 s
##     are recorded at full rate; the 4 s / 8 s gaps are recorded at a reduced
##     rate (seed-fixture-only expedient — every frame still carries its true
##     tick/timestamp_s so unseen_seconds is exact). See dataset_manifest.md.
##
## Run: godot --rendering-driver vulkan --script res://tests/dataset/seed_record.gd

const H := preload("res://tests/structural/structural_harness.gd")
const DH := preload("res://tests/dataset/dataset_harness.gd")

const RATE := DH.TICK_RATE
const WINDOW := 160
const MARGIN := 12


func _initialize() -> void:
	var root := DH.dataset_root()
	var sid := DH.new_session_id()
	var sc: Dictionary = await DH.build_scene(self, true)
	await process_frame
	var adapter := H.make_adapter(self, sid)
	var ex := H.make_exporter(self, sc.cam, sc.world, adapter, sid)
	var rec := DH.make_recorder(self, sid, root, ex, adapter, sc.cam)
	if not rec.begin():
		printerr("HARNESS_RESULT FAIL (recorder.begin refused)")
		quit(1)
		return

	var cam_home: Transform3D = sc.cam.global_transform
	var target_home: Vector3 = sc.e0.global_position
	var S := {
		"rec": rec, "rec_adapter": adapter, "tick": 0,
		"cam": sc.cam, "cam_home": cam_home,
		"target": sc.e0, "target_home": target_home,
		"occluder_pos": Vector3(1000, 0, -6),
		"full_rate_max": DH.FULL_RATE_MAX_TICKS,
	}

	# ---- temporal window: dense, contiguous, with motion ----
	var t0: int = rec.frame_count()
	for i in WINDOW:
		S.tick += 1
		var u := float(i) / float(WINDOW - 1)
		var yaw := lerpf(-18.0, 18.0, u)
		var t := cam_home
		t.basis = Basis(Vector3.UP, deg_to_rad(yaw)) * cam_home.basis
		sc.cam.global_transform = t
		sc.e0.global_position = target_home + Vector3(sin(u * TAU) * 1.5, 0.0, 0.0)
		await rec.record_frame(S.tick)
	rec.add_temporal_sequence("temporal_0", t0, int(rec.frame_count()) - 1)

	# ---- reappearance ladder: both mechanisms, all five rungs ----
	for mech in ["camera_away", "entity_away"]:
		for s in DH.RUNGS_SECONDS:
			var gap := int(round(s * RATE))
			var name := "reap_%s_%s" % [mech, _tag(s)]
			var seq: Dictionary = await DH.run_rung(S, name, gap, mech, MARGIN, MARGIN)
			print("  rung %-28s last_seen=%d first_again=%d unseen_frames=%d unseen_seconds=%.4f id_stable=%s"
				% [name, seq.last_seen_frame, seq.first_seen_again_frame,
					seq.unseen_frames, seq.unseen_seconds, seq.identity_stable])

	rec.finish()

	# ---- summary ----
	var sdir: String = rec.session_dir()
	var bytes := 0
	var nfiles := 0
	for f in DirAccess.get_files_at("%s/frames" % sdir):
		bytes += FileAccess.get_file_as_bytes("%s/frames/%s" % [sdir, f]).size()
		nfiles += 1
	print("SEED session_id : ", sid)
	print("SEED dir        : ", sdir)
	print("SEED frames     : ", rec.frame_count(), "  (", nfiles, " files in frames/, ",
		String.humanize_size(bytes), ")")
	print(FileAccess.get_file_as_string("%s/manifest.json" % sdir))
	print("HARNESS_RESULT PASS")
	quit(0)


func _tag(s: float) -> String:
	return ("%.1fs" % s).replace(".0s", "s")
