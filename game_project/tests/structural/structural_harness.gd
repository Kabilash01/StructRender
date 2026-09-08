extends RefCounted
## Shared helpers for the Milestone 1 structural-export verification scenes
## (S1-S6). Mirrors tests/harness.gd but for GPU structural export: these run
## via `godot --rendering-driver vulkan --script res://tests/structural/sN.gd`
## (NOT --headless — the dummy driver cannot render). See scripts/test_structural.sh.
##
## Loaded with preload (no class_name), same as the M0 harness.

const Exporter := preload("res://instrumentation/structural_exporter.gd")
const Adapter := preload("res://instrumentation/entity_lifecycle_adapter.gd")

const ENTITY_GROUP := &"structrender_entity"

static var _checks: int = 0
static var _failures: int = 0


static func check(cond: bool, msg: String) -> void:
	_checks += 1
	if cond:
		print("  ok   - ", msg)
	else:
		_failures += 1
		printerr("  FAIL - ", msg)


static func check_near(actual: float, expected: float, tol: float, msg: String) -> void:
	check(absf(actual - expected) <= tol,
		"%s (expected %.5f ± %.5f, got %.5f, |Δ|=%.5f)" % [msg, expected, tol, actual, absf(actual - expected)])


static func check_vec_near(actual: Vector3, expected: Vector3, tol: float, msg: String) -> void:
	check((actual - expected).length() <= tol,
		"%s (expected %v ± %.4f, got %v)" % [msg, expected, tol, actual])


static func finish(tree: SceneTree) -> void:
	if _failures == 0:
		print("HARNESS_RESULT PASS (%d checks)" % _checks)
		tree.quit(0)
	else:
		printerr("HARNESS_RESULT FAIL (%d checks, %d failure(s))" % [_checks, _failures])
		tree.quit(1)


static func make_world(tree: SceneTree) -> Node3D:
	Engine.set_meta(&"structrender_no_sim", true)
	var w := Node3D.new()
	w.name = "SynthWorld"
	tree.root.add_child(w)
	return w


## Author BOTH the structural-albedo meta and a real material from one linear
## colour, so the structural buffer and the NATIVE RGB describe the same scene.
##
## They used to disagree for every mesh (KB Decision 38): only the meta was set,
## so the structural albedo pass reported the authored colour while the native
## render — the sole image the teacher consumes, captured before any
## `material_override` swap — showed Godot's default grey. A cube authored
## (0.90, 0.65, 0.20) measured (0.66, 0.66, 0.66) in the recorded native RGB.
##
## The meta stays LINEAR and still wins for the structural pass
## (`frame_structure.md` §2.3 precedence); the material takes the sRGB form
## because `albedo_color` is interpreted as sRGB.
static func _apply_albedo(mi: MeshInstance3D, albedo: Color) -> void:
	mi.set_meta("structrender_albedo", albedo)
	var mat := StandardMaterial3D.new()
	mat.albedo_color = albedo.linear_to_srgb()
	mi.material_override = mat


## An entity: a Node3D in the structrender_entity group with a box MeshInstance
## child. `size` in metres, `albedo` interpreted linear.
static func add_entity(parent: Node3D, ename: String, pos: Vector3,
		size: float = 1.0, albedo: Color = Color(0.8, 0.8, 0.8)) -> Node3D:
	var e := Node3D.new()
	e.name = ename
	parent.add_child(e)
	e.add_to_group(ENTITY_GROUP)
	e.global_position = pos
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(size, size, size)
	mi.mesh = bm
	_apply_albedo(mi, albedo)
	e.add_child(mi)
	return e


## Non-entity static geometry (id 0), e.g. a backdrop wall.
## `rot_deg` lets a prop be tilted so it contributes a normal discontinuity as
## well as a depth jump (KB Decision 38 change 3).
static func add_prop(parent: Node3D, pos: Vector3, size: Vector3,
		albedo: Color = Color(0.5, 0.5, 0.5),
		rot_deg: Vector3 = Vector3.ZERO) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mi.mesh = bm
	_apply_albedo(mi, albedo)
	parent.add_child(mi)
	mi.global_position = pos
	if rot_deg != Vector3.ZERO:
		mi.rotation_degrees = rot_deg
	return mi


static func make_camera(parent: Node3D, from: Vector3, look_at: Vector3 = Vector3.INF) -> Camera3D:
	var c := Camera3D.new()
	parent.add_child(c)
	c.global_position = from
	if look_at != Vector3.INF:
		c.look_at(look_at, Vector3.UP)
	return c


static func make_exporter(tree: SceneTree, cam: Camera3D, scene_root: Node3D,
		adapter: Node, session_id: String = "session_test") -> Node:
	var ex := Exporter.new()
	tree.root.add_child(ex)
	ex.session_id = session_id
	ex.setup(cam, scene_root, adapter)
	return ex


static func make_adapter(tree: SceneTree, session_id: String = "session_test") -> Node:
	var a := Adapter.new()
	tree.root.add_child(a)
	a.session_id = session_id
	return a


static func px_index(px: Vector2, w: int) -> int:
	return int(px.y) * w + int(px.x)


## Bilinear-free nearest sample of a flat [H*W*C] buffer at UV (Y-down).
static func sample_c(buf: PackedFloat32Array, uv: Vector2, w: int, h: int, c: int, nc: int) -> float:
	var x := clampi(int(uv.x * w), 0, w - 1)
	var y := clampi(int(uv.y * h), 0, h - 1)
	return buf[(y * w + x) * nc + c]
