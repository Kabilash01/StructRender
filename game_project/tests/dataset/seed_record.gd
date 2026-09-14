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

	# ---- session variant (KB Decision 61) --------------------------------------
	# Variant 0 reproduces the committed session EXACTLY, so ds_1788687350_337657ea
	# stays regenerable. Variants > 0 perturb the CAMERA PATH and ENTITY
	# TRAJECTORIES only. Prop layout is deliberately NOT jittered: Decision 38's
	# three placement constraints (cover both camera poses, never fully occlude the
	# camera->target sightline, sit 3-40 m) are hand-verified for the current
	# layout, and the occlusion guard `_check_one_hidden_run` runs in the CI
	# fixture, not here. So this buys VIEWPOINT and TRAJECTORY diversity, not new
	# scene content, and the KB says so rather than implying more.
	var variant := int(OS.get_environment("STRUCTRENDER_SESSION_VARIANT"))
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260912 + variant * 1000
	var yaw_lo := -18.0
	var yaw_hi := 18.0
	var drift_amp := 1.5
	var drift_phase := 0.0
	var cam_home: Transform3D = sc.cam.global_transform
	if variant > 0:
		# Yaw the whole camera rig, so a different part of the enriched set is in
		# frame. Kept well inside the 165-deg aimed-away pose the rungs rely on.
		var rig_yaw := rng.randf_range(-30.0, 30.0)
		var t0v := cam_home
		t0v.basis = Basis(Vector3.UP, deg_to_rad(rig_yaw)) * cam_home.basis
		t0v.origin += Vector3(rng.randf_range(-1.2, 1.2), rng.randf_range(-0.3, 0.6), rng.randf_range(-1.0, 1.0))
		sc.cam.global_transform = t0v
		cam_home = t0v
		yaw_lo = rng.randf_range(-26.0, -10.0)
		yaw_hi = rng.randf_range(10.0, 26.0)
		drift_amp = rng.randf_range(0.9, 2.2)
		drift_phase = rng.randf_range(0.0, TAU)
		print("SEED variant    : ", variant, "  rig_yaw=", rig_yaw,
			"  sweep=[", yaw_lo, ",", yaw_hi, "]  drift_amp=", drift_amp)
	var target_home: Vector3 = sc.e0.global_position
	var S := {
		"rec": rec, "rec_adapter": adapter, "tick": 0,
		"cam": sc.cam, "cam_home": cam_home,
		"target": sc.e0, "target_home": target_home,
		# the second entity drifts on its own phase so the ladder frames carry
		# independent motion rather than repeating one pose (KB Decision 33)
		"target2": sc.e1, "target2_home": (sc.e1.global_position if sc.e1 != null else Vector3.ZERO),
		"occluder_pos": Vector3(1000, 0, -6),
		"full_rate_max": DH.FULL_RATE_MAX_TICKS,
	}

	# ---- temporal window: dense, contiguous, with motion ----
	var t0: int = rec.frame_count()
	for i in WINDOW:
		S.tick += 1
		var u := float(i) / float(WINDOW - 1)
		var yaw := lerpf(yaw_lo, yaw_hi, u)
		var t := cam_home
		t.basis = Basis(Vector3.UP, deg_to_rad(yaw)) * cam_home.basis
		sc.cam.global_transform = t
		sc.e0.global_position = target_home + Vector3(sin(u * TAU + drift_phase) * drift_amp, 0.0, 0.0)
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
