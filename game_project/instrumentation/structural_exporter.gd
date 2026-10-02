extends Node
## Milestone 1 — Structural Export Module (KB §4).
##
## Drives a half-float SubViewport that renders the active world several times
## with per-pass override shaders, reads each pass back with get_image(), and
## assembles a FrameStructure:
##   depth, normal, albedo_or_material, motion, entity_id_map,
##   camera_view, camera_projection, entity list.
##
## Motion is computed analytically from engine transforms per KB Decision 25
## (backward / screen-UV / Y-down / camera+object). Never pixel-estimated.
##
## Boundaries (KB §4): no silent zero-fill — a missing/failed channel aborts the
## frame loudly (see _validate). Entity IDs are addresses, not continuous input.
##
## Activation (KB Decision 24): no _process hooks; a test / the M2 recorder adds
## this node and calls export_frame() explicitly. Requires a real rendering
## device (GPU) — NOT --headless dummy. See scripts/test_structural.sh.
##
## Loaded via preload (no class_name), matching the M0 harness pattern.

const Shaders := preload("res://instrumentation/structural_shaders.gd")
const NpyWriter := preload("res://instrumentation/npy_writer.gd")

const SCHEMA_VERSION := "m1.0"
const ENTITY_GROUP := &"structrender_entity"
const NORMAL_LEN_TOL := 0.1

var width: int = 512
var height: int = 288
var session_id: String = "session_dev"

## S6 seam: set to "depth" | "normal" | "albedo" | "entity_id" to simulate that
## GPU pass failing (returns an all-zero image) so _validate must abort.
var debug_disable_pass: String = ""

## Diagnostic from the most recent aborted frame (channel-named). Empty on success.
var last_error: String = ""

var _viewport: SubViewport
var _vcam: Camera3D
var _src_cam: Camera3D
var _scene_root: Node3D
var _adapter: Node

var _depth_mat: ShaderMaterial
var _normal_mat: ShaderMaterial
var _albedo_shader: Shader
var _id_shader: Shader
var _albedo_mats: Dictionary = {}   # mesh -> ShaderMaterial
var _id_mats: Dictionary = {}       # mesh -> ShaderMaterial

# previous-frame state for motion
var _prev_valid: bool = false
var _prev_cam_xform: Transform3D
var _prev_proj: Projection
var _prev_entity_xforms: Dictionary = {}   # instance_id -> Transform3D
var _curr_entity_xforms: Dictionary = {}


func setup(source_camera: Camera3D, scene_root: Node3D, adapter: Node, res: Vector2i = Vector2i(512, 288)) -> void:
	_src_cam = source_camera
	_scene_root = scene_root
	_adapter = adapter
	width = res.x
	height = res.y

	_viewport = SubViewport.new()
	_viewport.size = Vector2i(width, height)
	_viewport.msaa_3d = Viewport.MSAA_DISABLED
	_viewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	_viewport.use_taa = false
	_viewport.use_hdr_2d = true
	_viewport.positional_shadow_atlas_size = 0
	_viewport.world_3d = source_camera.get_world_3d()
	_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(_viewport)

	_vcam = Camera3D.new()
	_viewport.add_child(_vcam)

	# Pure-black, ambient-off environment so background renders exactly (0,0,0)
	# and the depth pass can mark "no geometry" as eye-Z <= 0.
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0, 0, 0, 1)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_DISABLED
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	env.tonemap_exposure = 1.0
	env.glow_enabled = false
	env.fog_enabled = false
	_vcam.environment = env

	_depth_mat = ShaderMaterial.new()
	_depth_mat.shader = Shaders.make(Shaders.DEPTH)
	_normal_mat = ShaderMaterial.new()
	_normal_mat.shader = Shaders.make(Shaders.NORMAL)
	_albedo_shader = Shaders.make(Shaders.ALBEDO)
	_id_shader = Shaders.make(Shaders.ENTITY_ID)


## Returns a FrameStructure dict, or null if the frame failed validation
## (nothing is emitted in that case — KB "fail loudly, never zero-fill").
func export_frame(frame_id: int) -> Dictionary:
	assert(_viewport != null, "StructuralExporter.setup() not called")
	last_error = ""

	_adapter.scan(frame_id)
	_sync_camera()

	var meshes := _scene_root.find_children("*", "MeshInstance3D", true, false)
	var originals: Dictionary = {}
	for m in meshes:
		originals[m] = m.material_override

	# Map each mesh -> owning entity instance_id (0 = non-entity geometry).
	var mesh_ids: Dictionary = {}
	for m in meshes:
		var iid := int(_adapter.instance_id_for(m))
		if iid < 0:
			_restore(originals)
			return _abort("entity_id", "mesh '%s' belongs to an unregistered entity" % m.name)
		mesh_ids[m] = iid

	# ---- pass: depth ----
	for m in meshes:
		m.material_override = _depth_mat
	var depth_img := await _render_pass("depth")

	# ---- pass: normal ----
	for m in meshes:
		m.material_override = _normal_mat
	var normal_img := await _render_pass("normal")

	# ---- pass: albedo ----
	for m in meshes:
		m.material_override = _albedo_mat_for(m)
	var albedo_img := await _render_pass("albedo")

	# ---- pass: entity id ----
	for m in meshes:
		m.material_override = _id_mat_for(m, mesh_ids[m])
	var id_img := await _render_pass("entity_id")

	_restore(originals)

	# ---- assemble ----
	var n := width * height
	var depth := PackedFloat32Array()
	depth.resize(n)
	var normal := PackedFloat32Array()
	normal.resize(n * 3)
	var albedo := PackedFloat32Array()
	albedo.resize(n * 3)
	var id_map := PackedInt32Array()
	id_map.resize(n)
	var pixel_counts: Dictionary = {}

	var registered := _registered_ids()

	for y in height:
		for x in width:
			var pi := y * width + x
			var dr := depth_img.get_pixel(x, y).r
			var is_bg := dr <= 0.0
			depth[pi] = INF if is_bg else dr

			if is_bg:
				id_map[pi] = 0
				continue

			var nc := normal_img.get_pixel(x, y)
			normal[pi * 3] = nc.r
			normal[pi * 3 + 1] = nc.g
			normal[pi * 3 + 2] = nc.b

			var ac := albedo_img.get_pixel(x, y)
			albedo[pi * 3] = ac.r
			albedo[pi * 3 + 1] = ac.g
			albedo[pi * 3 + 2] = ac.b

			var ic := id_img.get_pixel(x, y)
			var eid := int(roundf(ic.r)) + int(roundf(ic.g)) * 256 + int(roundf(ic.b)) * 65536
			if eid != 0 and not registered.has(eid):
				return _abort("entity_id", "pixel (%d,%d) decoded unregistered instance_id %d" % [x, y, eid])
			id_map[pi] = eid
			if eid != 0:
				pixel_counts[eid] = int(pixel_counts.get(eid, 0)) + 1

	var view := _vcam.global_transform.affine_inverse()
	var proj := _vcam.get_camera_projection()
	var cam_view := _transform_rows(view)
	var cam_proj := _projection_rows(proj)

	_collect_entity_transforms()
	var motion_valid := _prev_valid
	var motion := _compute_motion(depth, id_map)

	_adapter.update_visibility(pixel_counts, frame_id)

	var entities: Array = _adapter.alive_entities()

	var fs := {
		"schema_version": SCHEMA_VERSION,
		"session_id": session_id,
		"frame_id": frame_id,
		"resolution": [width, height],
		"depth": depth,
		"normal": normal,
		"albedo_or_material": albedo,
		"motion": motion,
		"motion_valid": motion_valid,
		"entity_id_map": id_map,
		"camera_view": cam_view,
		"camera_projection": cam_proj,
		"camera_near": _vcam.near,
		"camera_far": _vcam.far,
		"entities": entities,
	}

	if not _validate(fs):
		return {}

	# stash previous-frame state for the next motion computation
	_prev_valid = true
	_prev_cam_xform = _vcam.global_transform
	_prev_proj = proj
	_prev_entity_xforms = _curr_entity_xforms.duplicate()

	return fs


## Buffer-space (width x height) pixel for a world point, via the export camera.
func project_point(world: Vector3) -> Vector2:
	return _vcam.unproject_position(world)


func vcam() -> Camera3D:
	return _vcam


func _render_pass(pass_name: String) -> Image:
	# two frames so the override + UPDATE_ALWAYS is guaranteed to have drawn
	await get_tree().process_frame
	await get_tree().process_frame
	await RenderingServer.frame_post_draw
	if debug_disable_pass == pass_name:
		var blank := Image.create(width, height, false, Image.FORMAT_RGBAH)
		blank.fill(Color(0, 0, 0, 1))
		return blank
	return _viewport.get_texture().get_image()


func _albedo_mat_for(mesh: MeshInstance3D) -> ShaderMaterial:
	if _albedo_mats.has(mesh):
		return _albedo_mats[mesh]
	var mat := ShaderMaterial.new()
	mat.shader = _albedo_shader
	mat.set_shader_parameter("base_color", _base_color_of(mesh))
	_albedo_mats[mesh] = mat
	return mat


func _id_mat_for(mesh: MeshInstance3D, iid: int) -> ShaderMaterial:
	var mat: ShaderMaterial = _id_mats.get(mesh, null)
	if mat == null:
		mat = ShaderMaterial.new()
		mat.shader = _id_shader
		_id_mats[mesh] = mat
	mat.set_shader_parameter("enc_r", float(iid & 255))
	mat.set_shader_parameter("enc_g", float((iid >> 8) & 255))
	mat.set_shader_parameter("enc_b", float((iid >> 16) & 255))
	return mat


func _base_color_of(mesh: MeshInstance3D) -> Vector3:
	if mesh.has_meta("structrender_albedo"):
		var c: Color = mesh.get_meta("structrender_albedo")
		return Vector3(c.r, c.g, c.b)  # meta is authored linear
	var mat := mesh.get_active_material(0)
	if mat is StandardMaterial3D:
		var lc := (mat as StandardMaterial3D).albedo_color.srgb_to_linear()
		return Vector3(lc.r, lc.g, lc.b)
	return Vector3(1, 1, 1)


func _sync_camera() -> void:
	_vcam.global_transform = _src_cam.global_transform
	_vcam.projection = _src_cam.projection
	_vcam.fov = _src_cam.fov
	_vcam.size = _src_cam.size
	_vcam.near = _src_cam.near
	_vcam.far = _src_cam.far
	_vcam.keep_aspect = _src_cam.keep_aspect
	_vcam.h_offset = _src_cam.h_offset
	_vcam.v_offset = _src_cam.v_offset
	_vcam.current = true


func _registered_ids() -> Dictionary:
	var s: Dictionary = {}
	for e in _adapter.alive_entities():
		s[int(e.instance_id)] = true
	return s


func _collect_entity_transforms() -> void:
	_curr_entity_xforms = {}
	for node in get_tree().get_nodes_in_group(ENTITY_GROUP):
		if not is_instance_valid(node) or not (node is Node3D):
			continue
		var iid := int(_adapter.instance_id_for(node))
		if iid > 0:
			_curr_entity_xforms[iid] = (node as Node3D).global_transform


func _compute_motion(depth: PackedFloat32Array, id_map: PackedInt32Array) -> PackedFloat32Array:
	var m := PackedFloat32Array()
	m.resize(width * height * 2)  # zero-filled
	if not _prev_valid:
		return m
	var prev_vp := _prev_proj * Projection(_prev_cam_xform.affine_inverse())
	for y in height:
		for x in width:
			var pi := y * width + x
			var d := depth[pi]
			if is_inf(d):
				continue  # background: parallax undefined -> 0 (documented)
			var eid := id_map[pi]
			if eid != 0 and not _prev_entity_xforms.has(eid):
				continue  # entity has no t-1 transform (spawned/registered this frame): no valid previous correspondence -> 0 (documented, like frame 0)
			var world_p := _vcam.project_position(Vector2(x + 0.5, y + 0.5), d)
			var mt: Transform3D = _curr_entity_xforms.get(eid, Transform3D.IDENTITY)
			var mtm1: Transform3D = _prev_entity_xforms.get(eid, Transform3D.IDENTITY)
			var world_p0: Vector3 = (mtm1 * mt.affine_inverse()) * world_p
			var clip := prev_vp * Vector4(world_p0.x, world_p0.y, world_p0.z, 1.0)
			if absf(clip.w) < 1e-9:
				continue
			var ndc := Vector3(clip.x, clip.y, clip.z) / clip.w
			var uv_prev_x := (ndc.x + 1.0) * 0.5
			var uv_prev_y := (1.0 - ndc.y) * 0.5
			var uv_curr_x := (x + 0.5) / float(width)
			var uv_curr_y := (y + 0.5) / float(height)
			m[pi * 2] = uv_prev_x - uv_curr_x
			m[pi * 2 + 1] = uv_prev_y - uv_curr_y
	return m


func _transform_rows(t: Transform3D) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(16)
	var cols := [t.basis.x, t.basis.y, t.basis.z]  # Basis .x/.y/.z are the columns
	for r in 3:
		for c in 3:
			out[r * 4 + c] = cols[c][r]
		out[r * 4 + 3] = t.origin[r]
	out[12] = 0.0
	out[13] = 0.0
	out[14] = 0.0
	out[15] = 1.0
	return out


func _projection_rows(p: Projection) -> PackedFloat32Array:
	var cols := [p.x, p.y, p.z, p.w]  # each a Vector4 column
	var out := PackedFloat32Array()
	out.resize(16)
	for c in 4:
		var col: Vector4 = cols[c]
		for r in 4:
			out[r * 4 + c] = col[r]
	return out


func _validate(fs: Dictionary) -> bool:
	var n := width * height

	if fs.depth.size() != n:
		_abort("depth", "size %d != %d" % [fs.depth.size(), n]); return false
	if fs.normal.size() != n * 3:
		_abort("normal", "size %d != %d" % [fs.normal.size(), n * 3]); return false
	if fs.albedo_or_material.size() != n * 3:
		_abort("albedo", "size %d != %d" % [fs.albedo_or_material.size(), n * 3]); return false
	if fs.motion.size() != n * 2:
		_abort("motion", "size %d != %d" % [fs.motion.size(), n * 2]); return false
	if fs.entity_id_map.size() != n:
		_abort("entity_id", "size %d != %d" % [fs.entity_id_map.size(), n]); return false
	if fs.camera_view.size() != 16 or fs.camera_projection.size() != 16:
		_abort("camera", "matrix not 4x4"); return false

	var geom_pixels := 0
	var depth_finite_sum := 0.0
	for pi in n:
		var d: float = fs.depth[pi]
		if is_inf(d):
			continue
		if is_nan(d) or d <= 0.0:
			_abort("depth", "non-finite / non-positive depth %f at index %d" % [d, pi]); return false
		geom_pixels += 1
		depth_finite_sum += d
		var nx: float = fs.normal[pi * 3]
		var ny: float = fs.normal[pi * 3 + 1]
		var nz: float = fs.normal[pi * 3 + 2]
		if is_nan(nx) or is_nan(ny) or is_nan(nz):
			_abort("normal", "NaN normal at index %d" % pi); return false
		var nlen := sqrt(nx * nx + ny * ny + nz * nz)
		if absf(nlen - 1.0) > NORMAL_LEN_TOL:
			_abort("normal", "normal length %f (not unit) at index %d" % [nlen, pi]); return false
		for k in 3:
			if is_nan(fs.albedo_or_material[pi * 3 + k]) or is_inf(fs.albedo_or_material[pi * 3 + k]):
				_abort("albedo", "non-finite albedo at index %d" % pi); return false

	if geom_pixels == 0:
		_abort("depth", "no geometry pixels — every pass would be background (pass failure?)"); return false

	for v in fs.camera_view:
		if is_nan(v) or is_inf(v):
			_abort("camera", "non-finite camera_view"); return false
	for v in fs.camera_projection:
		if is_nan(v) or is_inf(v):
			_abort("camera", "non-finite camera_projection"); return false

	for mi in fs.motion:
		if is_nan(mi) or is_inf(mi):
			_abort("motion", "non-finite motion component"); return false

	return true


func _restore(originals: Dictionary) -> void:
	for m in originals:
		if is_instance_valid(m):
			m.material_override = originals[m]


func _abort(channel: String, why: String) -> Dictionary:
	last_error = "channel '%s': %s" % [channel, why]
	push_error("StructuralExporter: channel '%s' FAILED — %s. Frame aborted, nothing emitted." % [channel, why])
	return {}


# ---------------------------------------------------------------------------
# Debug dumps (synthetic scenes only, M1). NPY for numeric assertions / M2,
# camera.json for the matrix convention. PNG previews: scripts/visualize_frame.py
# ---------------------------------------------------------------------------
func dump_frame(fs: Dictionary, dir: String) -> void:
	DirAccess.make_dir_recursive_absolute(dir)
	var base := "%s/frame_%d" % [dir, int(fs.frame_id)]
	NpyWriter.save_float32(base + "_depth.npy", fs.depth, [height, width, 1])
	NpyWriter.save_float32(base + "_normal.npy", fs.normal, [height, width, 3])
	NpyWriter.save_float32(base + "_albedo.npy", fs.albedo_or_material, [height, width, 3])
	NpyWriter.save_float32(base + "_motion.npy", fs.motion, [height, width, 2])
	NpyWriter.save_int32(base + "_entity_id.npy", fs.entity_id_map, [height, width])

	var cam := {
		"frame_id": fs.frame_id,
		"session_id": fs.session_id,
		"resolution": fs.resolution,
		"motion_valid": fs.motion_valid,
		"camera_near": fs.camera_near,
		"camera_far": fs.camera_far,
		"camera_view": Array(fs.camera_view),
		"camera_projection": Array(fs.camera_projection),
		"entities": fs.entities,
	}
	var f := FileAccess.open(base + "_camera.json", FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(cam, "  "))
		f.close()


## LIVE path (M10, KB Decision 86). The three passes a structure-only renderer
## reads — depth, normal, albedo — with the same camera sync and the same pass
## materials as export_frame(), returned as raw Images for StructRenderer to
## decode in C++. No entity-id pass, no motion, no per-pixel GDScript loop.
## export_frame() is untouched: it remains the recording path the dataset's
## determinism checks guard.
func capture_live() -> Array:
	assert(_viewport != null, "StructuralExporter.setup() not called")
	_sync_camera()
	var meshes := _scene_root.find_children("*", "MeshInstance3D", true, false)
	var originals: Dictionary = {}
	for m in meshes:
		originals[m] = m.material_override
	for m in meshes:
		m.material_override = _depth_mat
	var depth_img := await _render_pass("depth")
	for m in meshes:
		m.material_override = _normal_mat
	var normal_img := await _render_pass("normal")
	for m in meshes:
		m.material_override = _albedo_mat_for(m)
	var albedo_img := await _render_pass("albedo")
	_restore(originals)
	return [depth_img, normal_img, albedo_img]


## LIVE TEMPORAL path (KB Decision 97): capture_live()'s three passes plus the
## entity-id pass, with the camera and every registered entity's transform at
## this instant — everything StructRenderer.render_temporal() needs to compute
## motion and the spawned-entity mask in C++, as _compute_motion does here.
## Returns [depth, normal, albedo, ids, cam_xform, cam_proj, {iid: Transform3D}].
## export_frame() and capture_live() are untouched.
func capture_live_temporal(frame_id: int) -> Array:
	assert(_viewport != null, "StructuralExporter.setup() not called")
	_adapter.scan(frame_id)
	_sync_camera()
	var meshes := _scene_root.find_children("*", "MeshInstance3D", true, false)
	var originals: Dictionary = {}
	var mesh_ids: Dictionary = {}
	for m in meshes:
		originals[m] = m.material_override
		mesh_ids[m] = maxi(int(_adapter.instance_id_for(m)), 0)
	for m in meshes:
		m.material_override = _depth_mat
	var depth_img := await _render_pass("depth")
	for m in meshes:
		m.material_override = _normal_mat
	var normal_img := await _render_pass("normal")
	for m in meshes:
		m.material_override = _albedo_mat_for(m)
	var albedo_img := await _render_pass("albedo")
	for m in meshes:
		m.material_override = _id_mat_for(m, mesh_ids[m])
	var id_img := await _render_pass("entity_id")
	_restore(originals)
	_collect_entity_transforms()
	return [depth_img, normal_img, albedo_img, id_img, _vcam.global_transform,
		_vcam.get_camera_projection(), _curr_entity_xforms.duplicate()]
