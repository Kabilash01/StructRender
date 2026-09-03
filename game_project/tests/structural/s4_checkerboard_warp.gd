extends SceneTree
## S4 — checkerboard warp test (KB §12 "Motion Coordinate Convention" mitigation).
## A high-frequency piecewise-constant scene (grid of entity cubes). Backward-warp
## frame t-1 by the exported motion and assert it aligns with frame t.

const H := preload("res://tests/structural/structural_harness.gd")

const W_RES := 512
const H_RES := 288
const CAM_MOVE := Vector3(0.35, 0.22, 0.0)
const GRID_X := 9
const GRID_Y := 6
const GRID_Z := -8.0


func _initialize() -> void:
	var world := H.make_world(self)
	await process_frame

	# wall behind so there is no true background, then a dense grid of entity cubes
	H.add_prop(world, Vector3(0, 0, GRID_Z - 2.0), Vector3(200, 200, 0.5), Color(0.2, 0.2, 0.2))
	var idx := 0
	for gy in GRID_Y:
		for gx in GRID_X:
			var col := Color(0.15 + 0.09 * gx, 0.15 + 0.13 * gy, 0.6)
			H.add_entity(world, "C_%d" % idx, Vector3((gx - 4) * 1.3, (gy - 2.5) * 1.3, GRID_Z), 1.0, col)
			idx += 1
	var cam := H.make_camera(world, Vector3(0, 0, 0))
	await process_frame

	var adapter := H.make_adapter(self)
	var ex := H.make_exporter(self, cam, world, adapter)

	var fs0: Dictionary = await ex.export_frame(0)
	H.check(not fs0.is_empty(), "frame 0 exported")
	var id0: PackedInt32Array = fs0.entity_id_map
	var depth0: PackedFloat32Array = fs0.depth

	cam.global_position += CAM_MOVE
	await process_frame
	var fs1: Dictionary = await ex.export_frame(1)
	H.check(not fs1.is_empty() and fs1.motion_valid, "frame 1 exported, motion_valid")
	if fs1.is_empty():
		H.finish(self)
		return
	var id1: PackedInt32Array = fs1.entity_id_map
	var depth1: PackedFloat32Array = fs1.depth
	var motion: PackedFloat32Array = fs1.motion

	var margin := 8
	var total := 0
	var id_mismatch := 0
	var depth_err_sum := 0.0
	var depth_n := 0
	for y in range(margin, H_RES - margin):
		for x in range(margin, W_RES - margin):
			var pi := y * W_RES + x
			var uv_x := (x + 0.5) / float(W_RES) + motion[pi * 2]
			var uv_y := (y + 0.5) / float(H_RES) + motion[pi * 2 + 1]
			if uv_x < 0.0 or uv_x >= 1.0 or uv_y < 0.0 or uv_y >= 1.0:
				continue
			var sx := int(uv_x * W_RES)
			var sy := int(uv_y * H_RES)
			var si := sy * W_RES + sx
			total += 1
			if id0[si] != id1[pi]:
				id_mismatch += 1
			if not is_inf(depth0[si]) and not is_inf(depth1[pi]):
				depth_err_sum += absf(depth0[si] - depth1[pi])
				depth_n += 1

	var frac := float(id_mismatch) / float(max(total, 1))
	var mean_depth_err := depth_err_sum / float(max(depth_n, 1))
	print("  info - warped %d px, id mismatch %.3f%%, mean depth warp err %.4f m" % [total, frac * 100.0, mean_depth_err])
	H.check(frac < 0.05, "backward-warp id-map mismatch < 5%% (got %.2f%%)" % (frac * 100.0))
	H.check(mean_depth_err < 0.08, "backward-warp mean depth error < 0.08 m (got %.4f)" % mean_depth_err)

	ex.dump_frame(fs1, "user://structural_debug/s4")
	H.finish(self)
