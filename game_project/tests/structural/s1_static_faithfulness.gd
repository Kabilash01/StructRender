extends SceneTree
## S1 — static structural faithfulness (KB §11 Verification Scenario 1).
## Known camera, a unit cube at a known distance with a set base colour, plus
## 3 entities so the id map is exactly {0,1,2,3}. Asserts exported *values*.

const H := preload("res://tests/structural/structural_harness.gd")

const W_RES := 512
const H_RES := 288


func _initialize() -> void:
	var world := H.make_world(self)
	await process_frame

	# 3 entities; cube1 is the geometry/material subject at the origin.
	var cube1 := H.add_entity(world, "Cube1", Vector3(0, 0, 0), 1.0, Color(0.2, 0.6, 0.8))
	var cube2 := H.add_entity(world, "Cube2", Vector3(-3, 0, 0), 1.0, Color(0.7, 0.1, 0.1))
	var cube3 := H.add_entity(world, "Cube3", Vector3(3, 0, 0), 1.0, Color(0.1, 0.7, 0.1))
	var cam := H.make_camera(world, Vector3(4, 3, 6), Vector3.ZERO)
	await process_frame

	var adapter := H.make_adapter(self)
	var ex := H.make_exporter(self, cam, world, adapter)

	var fs: Dictionary = await ex.export_frame(0)
	H.check(not fs.is_empty(), "frame 0 exported (last_error='%s')" % ex.last_error)
	if fs.is_empty():
		H.finish(self)
		return

	var depth: PackedFloat32Array = fs.depth
	var normal: PackedFloat32Array = fs.normal
	var albedo: PackedFloat32Array = fs.albedo_or_material
	var id_map: PackedInt32Array = fs.entity_id_map
	var view_inv := cam.global_transform.affine_inverse()

	# ---- depth at cube1 +Z face centre ----
	var p_front := Vector3(0, 0, 0.5)
	var px_front: Vector2 = ex.project_point(p_front)
	var i_front := H.px_index(px_front, W_RES)
	var expected_depth := -(view_inv * p_front).z
	H.check_near(depth[i_front], expected_depth, 0.06, "depth at cube1 +Z face")

	# ---- background depth is the sentinel, never 0 ----
	var i_corner := 2 * W_RES + 2
	H.check(is_inf(depth[i_corner]), "background depth == +inf sentinel (got %f)" % depth[i_corner])
	H.check(depth[i_corner] != 0.0, "background depth != 0")

	# ---- world-space face normals == known axes, unit length ----
	_check_normal(normal, ex.project_point(Vector3(0, 0, 0.5)), Vector3(0, 0, 1), "cube1 +Z face normal")
	_check_normal(normal, ex.project_point(Vector3(0, 0.5, 0)), Vector3(0, 1, 0), "cube1 +Y face normal")
	_check_normal(normal, ex.project_point(Vector3(0.5, 0, 0)), Vector3(1, 0, 0), "cube1 +X face normal")

	# every non-background normal is unit length
	var bad_norm := 0
	for pi in W_RES * H_RES:
		if is_inf(depth[pi]):
			continue
		var n := Vector3(normal[pi * 3], normal[pi * 3 + 1], normal[pi * 3 + 2])
		if absf(n.length() - 1.0) > 0.05:
			bad_norm += 1
	H.check(bad_norm == 0, "all %d geometry normals unit-length (%d bad)" % [W_RES * H_RES, bad_norm])

	# ---- albedo == set linear colour ----
	var a := Vector3(albedo[i_front * 3], albedo[i_front * 3 + 1], albedo[i_front * 3 + 2])
	H.check_vec_near(a, Vector3(0.2, 0.6, 0.8), 0.02, "cube1 albedo (linear)")

	# ---- entity-id map is exactly {0,1,2,3}, int32, correct per entity ----
	H.check(id_map is PackedInt32Array, "entity_id_map is int32")
	var uniq := {}
	for v in id_map:
		uniq[v] = true
	var keys := uniq.keys()
	keys.sort()
	H.check(keys == [0, 1, 2, 3], "unique entity ids == [0,1,2,3] (got %s)" % str(keys))
	H.check(adapter.instance_id_for(cube1) == 1, "Cube1 instance_id == 1")
	H.check(adapter.instance_id_for(cube2) == 2, "Cube2 instance_id == 2")
	H.check(adapter.instance_id_for(cube3) == 3, "Cube3 instance_id == 3")
	H.check(id_map[i_front] == 1, "id map at Cube1 front face == 1 (got %d)" % id_map[i_front])
	# per-entity pixel presence
	var counts := {}
	for v in id_map:
		counts[v] = int(counts.get(v, 0)) + 1
	H.check(int(counts.get(1, 0)) > 100 and int(counts.get(2, 0)) > 100 and int(counts.get(3, 0)) > 100,
		"each entity owns a plausible pixel block (%d/%d/%d)" % [counts.get(1,0), counts.get(2,0), counts.get(3,0)])

	# ---- camera matrix: projection @ view @ world_pt matches unproject_position ----
	for p in [Vector3(0, 0, 0.5), Vector3(0.5, 0, 0), Vector3(-3, 0.5, 0), Vector3(3, 0, 0.5)]:
		var manual: Vector2 = _manual_project(fs.camera_projection, fs.camera_view, p)
		var godot_px: Vector2 = ex.project_point(p)
		H.check((manual - godot_px).length() <= 1.5,
			"cam matrix projects %v to %v (unproject_position=%v)" % [p, manual, godot_px])

	# ---- debug dumps for eyeballing ----
	ex.dump_frame(fs, "user://structural_debug/s1")
	adapter.flush_events("user://structural_debug/s1/events.jsonl")
	print("  info - dumped to user://structural_debug/s1 (", OS.get_user_data_dir(), ")")

	H.finish(self)


func _check_normal(normal: PackedFloat32Array, px: Vector2, expected_axis: Vector3, msg: String) -> void:
	var i := H.px_index(px, W_RES)
	var n := Vector3(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2])
	H.check_vec_near(n, expected_axis, 0.1, msg)
	H.check_near(n.length(), 1.0, 0.05, msg + " unit length")


func _manual_project(proj: PackedFloat32Array, view: PackedFloat32Array, p: Vector3) -> Vector2:
	var v := _mat_vec(view, [p.x, p.y, p.z, 1.0])
	var c := _mat_vec(proj, v)
	var ndc_x: float = c[0] / c[3]
	var ndc_y: float = c[1] / c[3]
	return Vector2((ndc_x + 1.0) * 0.5 * W_RES, (1.0 - ndc_y) * 0.5 * H_RES)


func _mat_vec(m: PackedFloat32Array, v: Array) -> Array:
	var out := [0.0, 0.0, 0.0, 0.0]
	for r in 4:
		var s := 0.0
		for col in 4:
			s += m[r * 4 + col] * v[col]
		out[r] = s
	return out
