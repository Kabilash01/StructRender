extends SceneTree
## S3 — object motion (KB §11 Verification Scenario, KB §12 "camera-only vs object motion").
## Static camera, one entity translates a known amount, everything else static.
## Asserts that entity's pixels show the analytic backward motion and that
## static background pixels are ~0.

const H := preload("res://tests/structural/structural_harness.gd")

const W_RES := 512
const H_RES := 288
const MOVER_DELTA := Vector3(1.2, 0.5, 0.0)   # metres, between t-1 and t


func _initialize() -> void:
	var world := H.make_world(self)
	await process_frame

	H.add_prop(world, Vector3(0, 0, -12), Vector3(200, 200, 0.5), Color(0.35, 0.35, 0.4))
	var mover := H.add_entity(world, "Mover", Vector3(-1.5, -0.5, -6), 1.5, Color(0.9, 0.5, 0.1))
	var still := H.add_entity(world, "Still", Vector3(3.0, 0.0, -6), 1.5, Color(0.1, 0.5, 0.9))
	var cam := H.make_camera(world, Vector3(0, 0, 0))
	await process_frame

	var adapter := H.make_adapter(self)
	var ex := H.make_exporter(self, cam, world, adapter)

	var fs0: Dictionary = await ex.export_frame(0)
	H.check(not fs0.is_empty(), "frame 0 exported")

	# advance: only the mover moves
	mover.global_position += MOVER_DELTA
	await process_frame
	var fs1: Dictionary = await ex.export_frame(1)
	H.check(not fs1.is_empty() and fs1.motion_valid, "frame 1 exported, motion_valid")
	if fs1.is_empty():
		H.finish(self)
		return

	var depth: PackedFloat32Array = fs1.depth
	var motion: PackedFloat32Array = fs1.motion
	var id_map: PackedInt32Array = fs1.entity_id_map
	var mover_id: int = adapter.instance_id_for(mover)
	var still_id: int = adapter.instance_id_for(still)

	# ---- moving entity pixels: analytic backward motion ----
	var checked_mover := 0
	var mover_max_err := 0.0
	var mover_signs_ok := true
	for sp in [Vector2(220, 150), Vector2(250, 130), Vector2(210, 170)]:
		var px_t: Vector2 = sp
		var it := int(px_t.y) * W_RES + int(px_t.x)
		if id_map[it] != mover_id:
			continue
		checked_mover += 1
		var world_p: Vector3 = ex.vcam().project_position(px_t + Vector2(0.5, 0.5), depth[it])
		var px_prev: Vector2 = ex.vcam().unproject_position(world_p - MOVER_DELTA)
		var exp_x := (px_prev.x - (px_t.x + 0.5)) / float(W_RES)
		var exp_y := (px_prev.y - (px_t.y + 0.5)) / float(H_RES)
		var err := Vector2(motion[it * 2] - exp_x, motion[it * 2 + 1] - exp_y).length()
		mover_max_err = maxf(mover_max_err, err)
		if motion[it * 2] >= 0.0:
			mover_signs_ok = false
		H.check(err < 0.006, "mover motion at %v matches analytic object motion (err=%.5f)" % [sp, err])
	H.check(checked_mover >= 2, "sampled at least 2 mover pixels (got %d)" % checked_mover)
	H.check(mover_signs_ok, "object +X => backward motion.x < 0 on mover (opposite of camera case)")

	# ---- static entity + background: motion ~ 0 ----
	var still_max := 0.0
	var bg_max := 0.0
	var n_still := 0
	var n_bg := 0
	for pi in W_RES * H_RES:
		var mx: float = motion[pi * 2]
		var my: float = motion[pi * 2 + 1]
		var mag := sqrt(mx * mx + my * my)
		if id_map[pi] == still_id:
			still_max = maxf(still_max, mag)
			n_still += 1
		elif id_map[pi] == 0 and not is_inf(depth[pi]):
			bg_max = maxf(bg_max, mag)
			n_bg += 1
	H.check(n_still > 200 and still_max < 0.001, "static entity motion ~0 (max=%.6f over %d px)" % [still_max, n_still])
	H.check(n_bg > 1000 and bg_max < 0.001, "static background motion ~0 (max=%.6f over %d px)" % [bg_max, n_bg])
	print("  info - mover max err %.5f UV ; still max %.6f ; bg max %.6f" % [mover_max_err, still_max, bg_max])

	ex.dump_frame(fs1, "user://structural_debug/s3")
	H.finish(self)
