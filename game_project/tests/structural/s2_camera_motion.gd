extends SceneTree
## S2 — camera motion (KB §11 Verification Scenario 2, KB §12 "known translation test").
## Static scene, camera translates a known +X between two frames. Asserts the
## exported backward motion field matches the closed-form screen displacement,
## with correct sign/direction.

const H := preload("res://tests/structural/structural_harness.gd")

const W_RES := 512
const H_RES := 288
const WALL_Z := -10.0
const CAM_Z := 0.0
const DX := 0.6   # metres, +X


func _initialize() -> void:
	var world := H.make_world(self)
	await process_frame

	# A big fronto-parallel wall so every pixel has geometry (no background).
	H.add_prop(world, Vector3(0, 0, WALL_Z), Vector3(200, 200, 0.5), Color(0.4, 0.4, 0.45))
	# a couple of entities on the wall for good measure
	H.add_entity(world, "E1", Vector3(-2, 1, WALL_Z + 0.4), 1.0, Color(0.8, 0.2, 0.2))
	H.add_entity(world, "E2", Vector3(3, -1, WALL_Z + 0.4), 1.0, Color(0.2, 0.8, 0.2))
	var cam := H.make_camera(world, Vector3(0, 0, CAM_Z))  # looks down -Z
	await process_frame

	var adapter := H.make_adapter(self)
	var ex := H.make_exporter(self, cam, world, adapter)

	var fs0: Dictionary = await ex.export_frame(0)
	H.check(not fs0.is_empty() and fs0.motion_valid == false, "frame 0: motion_valid == false")

	# reference camera at the frame-0 pose in a correctly sized viewport
	var ref_vp := SubViewport.new()
	ref_vp.size = Vector2i(W_RES, H_RES)
	root.add_child(ref_vp)
	var ref_cam := Camera3D.new()
	ref_vp.add_child(ref_cam)
	await process_frame
	ref_cam.global_transform = cam.global_transform
	ref_cam.fov = cam.fov
	ref_cam.near = cam.near
	ref_cam.far = cam.far
	ref_cam.current = true

	# advance: move camera +X
	cam.global_position += Vector3(DX, 0, 0)
	await process_frame
	var fs1: Dictionary = await ex.export_frame(1)
	H.check(not fs1.is_empty() and fs1.motion_valid == true, "frame 1: motion_valid == true")
	if fs1.is_empty():
		H.finish(self)
		return

	var depth: PackedFloat32Array = fs1.depth
	var motion: PackedFloat32Array = fs1.motion

	# ---- closed form at the image centre (near optical axis) ----
	var cx := W_RES / 2
	var cy := H_RES / 2
	var ci := cy * W_RES + cx
	var d_centre := depth[ci]
	var tan_half_x: float = 1.0 / fs1.camera_projection[0]  # row-major [0][0] = 1/tan(haleffov_x)
	var expected_uv_x := 0.5 * DX / (tan_half_x * d_centre)
	H.check_near(motion[ci * 2], expected_uv_x, 0.004, "centre motion.x == closed form 0.5*dx/(tan*d)")
	H.check(motion[ci * 2] > 0.0, "camera +X => backward motion.x > 0 (sign)")
	H.check_near(motion[ci * 2 + 1], 0.0, 0.003, "centre motion.y ~ 0 (pure X translation)")

	# ---- reference-camera check at several sample points ----
	var max_err := 0.0
	for sp in [Vector2(180, 100), Vector2(300, 200), Vector2(256, 144), Vector2(400, 120)]:
		var px_t: Vector2 = sp
		var it := int(px_t.y) * W_RES + int(px_t.x)
		var world_p: Vector3 = ex.vcam().project_position(px_t + Vector2(0.5, 0.5), depth[it])
		var px_prev: Vector2 = ref_cam.unproject_position(world_p)
		var exp_x := (px_prev.x - (px_t.x + 0.5)) / float(W_RES)
		var exp_y := (px_prev.y - (px_t.y + 0.5)) / float(H_RES)
		var err := Vector2(motion[it * 2] - exp_x, motion[it * 2 + 1] - exp_y).length()
		max_err = maxf(max_err, err)
		H.check(err < 0.006, "motion at %v matches ref-cam reprojection (err=%.5f UV)" % [sp, err])
	print("  info - max reference-reprojection error: %.5f UV (~%.2f px)" % [max_err, max_err * W_RES])

	ex.dump_frame(fs1, "user://structural_debug/s2")
	H.finish(self)
