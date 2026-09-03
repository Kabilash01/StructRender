extends SceneTree
## S7 — motion for an entity first seen after frame 0
## (KB Decision 25 "no valid previous correspondence -> 0", spec-guardian follow-up).
##
## Frame 0: static camera + entity A visible.
## Frame 1: A translates a known amount AND a new entity B is spawned/registered
##          in view. B has NO frame-0 transform, so its pixels must be motion
##          (0, 0) EXACTLY (documented, like frame 0) — not a garbage vector from
##          M_{t-1} = identity. A's pixels must still show the correct analytic
##          object motion (the fix must not perturb existing-entity motion).
## Frame 2: B (now with a frame-1 transform) translates a known amount; its
##          pixels must recover the correct analytic motion.

const H := preload("res://tests/structural/structural_harness.gd")

const W_RES := 512
const H_RES := 288

const A_DELTA := Vector3(1.0, 0.4, 0.0)    # metres, A moves between t0 and t1
const B_DELTA := Vector3(-0.8, 0.3, 0.0)   # metres, B moves between t1 and t2


func _initialize() -> void:
	var world := H.make_world(self)
	await process_frame

	# a backdrop that fills the centre but leaves a sky border (depth == +inf)
	H.add_prop(world, Vector3(0, 0, -12), Vector3(16, 9, 0.5), Color(0.3, 0.3, 0.35))
	var a := H.add_entity(world, "A", Vector3(-1.6, 0.0, -6), 1.6, Color(0.9, 0.5, 0.1))
	var cam := H.make_camera(world, Vector3(0, 0, 0))   # static, looks down -Z
	await process_frame

	var adapter := H.make_adapter(self, "session_s7")
	var ex := H.make_exporter(self, cam, world, adapter, "session_s7")

	# ---- frame 0: only A exists ----
	var fs0: Dictionary = await ex.export_frame(0)
	H.check(not fs0.is_empty() and fs0.motion_valid == false, "frame 0 exported, motion_valid == false")

	# ---- advance to frame 1: A moves, B is spawned in view ----
	a.global_position += A_DELTA
	var b := H.add_entity(world, "B", Vector3(2.2, 0.0, -6), 1.6, Color(0.1, 0.6, 0.9))
	await process_frame

	var fs1: Dictionary = await ex.export_frame(1)
	H.check(not fs1.is_empty() and fs1.motion_valid == true, "frame 1 exported, motion_valid == true")
	if fs1.is_empty():
		H.finish(self)
		return

	var a_id: int = adapter.instance_id_for(a)
	var b_id: int = adapter.instance_id_for(b)
	H.check(b_id > 0 and b_id != a_id, "B registered with its own instance_id (a=%d b=%d)" % [a_id, b_id])

	# B must have been a genuine frame-1 spawn (no prior transform)
	var b_spawn_frame := -1
	for e in adapter.events():
		if e.instance_id == b_id and e.event == "spawn":
			b_spawn_frame = e.frame_id
	H.check(b_spawn_frame == 1, "B spawn event emitted at frame 1 (got %d)" % b_spawn_frame)

	var depth1: PackedFloat32Array = fs1.depth
	var motion1: PackedFloat32Array = fs1.motion
	var id1: PackedInt32Array = fs1.entity_id_map

	# ---- B's pixels: motion EXACTLY (0, 0) (no previous correspondence) ----
	var n_b := 0
	var b_nonzero := 0
	for pi in W_RES * H_RES:
		if id1[pi] != b_id:
			continue
		n_b += 1
		if motion1[pi * 2] != 0.0 or motion1[pi * 2 + 1] != 0.0:
			b_nonzero += 1
	H.check(n_b > 400, "B has a substantial pixel footprint in frame 1 (%d px)" % n_b)
	H.check(b_nonzero == 0, "frame 1: ALL %d of B's pixels have motion exactly (0,0) — no prior transform (nonzero: %d)" % [n_b, b_nonzero])

	# ---- A's pixels: correct analytic backward object motion (fix didn't perturb) ----
	var n_a := 0
	var a_max_err := 0.0
	var a_min_mag := INF
	for sp_v in [Vector2(225, 125), Vector2(240, 135), Vector2(230, 140), Vector2(245, 120)]:
		var sp: Vector2 = sp_v
		var it := int(sp.y) * W_RES + int(sp.x)
		if id1[it] != a_id:
			continue
		n_a += 1
		var world_p: Vector3 = ex.vcam().project_position(sp + Vector2(0.5, 0.5), depth1[it])
		var px_prev: Vector2 = ex.vcam().unproject_position(world_p - A_DELTA)
		var exp_x := (px_prev.x - (sp.x + 0.5)) / float(W_RES)
		var exp_y := (px_prev.y - (sp.y + 0.5)) / float(H_RES)
		var err := Vector2(motion1[it * 2] - exp_x, motion1[it * 2 + 1] - exp_y).length()
		a_max_err = maxf(a_max_err, err)
		a_min_mag = minf(a_min_mag, Vector2(exp_x, exp_y).length())
		H.check(err < 0.006, "frame 1: A motion at %v matches analytic object motion (err=%.5f UV)" % [sp, err])
	H.check(n_a >= 3, "sampled at least 3 of A's pixels in frame 1 (got %d)" % n_a)
	H.check(a_min_mag > 0.01, "frame 1: A's motion is non-trivially non-zero (min |m|=%.4f UV)" % a_min_mag)

	# ---- background (id 0): finite-depth backdrop ~0, sky (+inf) exactly 0 ----
	var bg_max := 0.0
	var n_bg_finite := 0
	var n_sky := 0
	var sky_nonzero := 0
	for pi in W_RES * H_RES:
		if id1[pi] != 0:
			continue
		if is_inf(depth1[pi]):
			n_sky += 1
			if motion1[pi * 2] != 0.0 or motion1[pi * 2 + 1] != 0.0:
				sky_nonzero += 1
		else:
			n_bg_finite += 1
			bg_max = maxf(bg_max, Vector2(motion1[pi * 2], motion1[pi * 2 + 1]).length())
	H.check(n_bg_finite > 1000 and bg_max < 0.001, "frame 1: static backdrop motion ~0 (max=%.6f over %d px)" % [bg_max, n_bg_finite])
	H.check(n_sky > 1000, "frame 1: sky (depth==+inf) pixels present (%d px)" % n_sky)
	H.check(sky_nonzero == 0, "frame 1: ALL %d sky pixels have motion exactly (0,0) (nonzero: %d)" % [n_sky, sky_nonzero])

	# ---- advance to frame 2: B now has a frame-1 transform, move it a known amount ----
	b.global_position += B_DELTA
	await process_frame

	var fs2: Dictionary = await ex.export_frame(2)
	H.check(not fs2.is_empty() and fs2.motion_valid == true, "frame 2 exported, motion_valid == true")
	if fs2.is_empty():
		H.finish(self)
		return

	var depth2: PackedFloat32Array = fs2.depth
	var motion2: PackedFloat32Array = fs2.motion
	var id2: PackedInt32Array = fs2.entity_id_map

	var n_b2 := 0
	var b2_max_err := 0.0
	var b2_min_mag := INF
	var b2_signs_ok := true
	for sp_v in [Vector2(290, 135), Vector2(300, 140), Vector2(285, 130), Vector2(305, 145)]:
		var sp: Vector2 = sp_v
		var it := int(sp.y) * W_RES + int(sp.x)
		if id2[it] != b_id:
			continue
		n_b2 += 1
		var world_p: Vector3 = ex.vcam().project_position(sp + Vector2(0.5, 0.5), depth2[it])
		var px_prev: Vector2 = ex.vcam().unproject_position(world_p - B_DELTA)
		var exp_x := (px_prev.x - (sp.x + 0.5)) / float(W_RES)
		var exp_y := (px_prev.y - (sp.y + 0.5)) / float(H_RES)
		var err := Vector2(motion2[it * 2] - exp_x, motion2[it * 2 + 1] - exp_y).length()
		b2_max_err = maxf(b2_max_err, err)
		b2_min_mag = minf(b2_min_mag, Vector2(exp_x, exp_y).length())
		# object moves -X between t1 and t2 => backward motion.x > 0
		if motion2[it * 2] <= 0.0:
			b2_signs_ok = false
		H.check(err < 0.006, "frame 2: B motion at %v matches analytic object motion (err=%.5f UV)" % [sp, err])
	H.check(n_b2 >= 3, "sampled at least 3 of B's pixels in frame 2 (got %d)" % n_b2)
	H.check(b2_min_mag > 0.01, "frame 2: B's recovered motion is non-trivially non-zero (min |m|=%.4f UV)" % b2_min_mag)
	H.check(b2_signs_ok, "frame 2: B moved -X => backward motion.x > 0 (sign)")

	print("  info - frame1: A max err %.5f UV, A min |m| %.4f ; B footprint %d px (all zero)" % [a_max_err, a_min_mag, n_b])
	print("  info - frame2: B max err %.5f UV, B min |m| %.4f" % [b2_max_err, b2_min_mag])

	ex.dump_frame(fs1, "user://structural_debug/s7/f1")
	ex.dump_frame(fs2, "user://structural_debug/s7/f2")
	H.finish(self)
