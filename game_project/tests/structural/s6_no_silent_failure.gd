extends SceneTree
## S6 — no silent channel failure (KB §4 Structural Export boundary:
## "does not fill missing buffers with zeros silently"; KB §10 "fail loudly offline").
##
## (a) a rendering pass fails  -> exporter aborts, names the channel, emits nothing.
## (b) an entity has no registered id -> exporter aborts, names the channel, emits nothing.
## In both cases: NO FrameStructure is returned and NO zero-filled buffer appears.

const H := preload("res://tests/structural/structural_harness.gd")

const DUMP_DIR := "user://structural_debug/s6"


func _initialize() -> void:
	var world := H.make_world(self)
	await process_frame
	H.add_prop(world, Vector3(0, 0, -10), Vector3(60, 60, 0.5), Color(0.3, 0.3, 0.3))
	var cube := H.add_entity(world, "Subject", Vector3(0, 0, -5), 1.5, Color(0.7, 0.4, 0.9))
	var cam := H.make_camera(world, Vector3(0, 0, 0))
	await process_frame

	var adapter := H.make_adapter(self)
	var ex := H.make_exporter(self, cam, world, adapter)

	# clear any stale dump dir
	_rm_dir(DUMP_DIR)

	# ---- control: a clean frame really does export ----
	var ok_fs: Dictionary = await ex.export_frame(0)
	H.check(not ok_fs.is_empty(), "control: clean frame exports (last_error='%s')" % ex.last_error)

	# ---- (a) simulate a failed GPU pass ----
	ex.debug_disable_pass = "normal"
	var fs_a: Dictionary = await ex.export_frame(1)
	H.check(fs_a.is_empty(), "S6a: aborted frame returns an EMPTY FrameStructure (no zero-fill)")
	H.check(fs_a.get("depth") == null and fs_a.get("normal") == null,
		"S6a: no buffers present in the aborted result")
	H.check(ex.last_error.contains("normal"),
		"S6a: diagnostic names the failing channel 'normal' (got '%s')" % ex.last_error)
	ex.debug_disable_pass = ""

	# ---- (b) an entity with no registered instance_id ----
	adapter.frozen = true                       # scan() becomes a no-op
	var ghost := Node3D.new()
	ghost.name = "GhostEntity"
	world.add_child(ghost)
	ghost.add_to_group(&"structrender_entity")  # in the group, but never scanned -> no record
	var gm := MeshInstance3D.new()
	gm.mesh = BoxMesh.new()
	ghost.add_child(gm)
	ghost.global_position = Vector3(1.5, 0, -5)
	await process_frame

	var fs_b: Dictionary = await ex.export_frame(2)
	H.check(fs_b.is_empty(), "S6b: aborted frame returns an EMPTY FrameStructure (no zero-fill)")
	H.check(ex.last_error.contains("entity_id") and ex.last_error.to_lower().contains("unregistered"),
		"S6b: diagnostic names channel 'entity_id' + 'unregistered' (got '%s')" % ex.last_error)

	# ---- no debug output was written for the aborted frames ----
	var leaked := _list_dir(DUMP_DIR)
	H.check(leaked.is_empty(), "S6: no debug files written for aborted frames (found %s)" % str(leaked))

	H.finish(self)


func _rm_dir(path: String) -> void:
	var d := DirAccess.open(path)
	if d == null:
		return
	for f in d.get_files():
		d.remove(f)


func _list_dir(path: String) -> PackedStringArray:
	var d := DirAccess.open(path)
	if d == null:
		return PackedStringArray()
	return d.get_files()
