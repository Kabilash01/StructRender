extends Node
## Milestone 1 — Entity Lifecycle Adapter (KB §4 "Entity Lifecycle Adapter",
## KB §12 "Entity ID Reuse").
##
## Converts Godot group membership + engine-index reuse + entity-ID-map pixel
## counts into stable StructRender lifecycle events:
##
##   {session_id, instance_id, generation, event, frame_id}   event in
##   {spawn, destroy, visible, hidden}
##
## Boundaries (KB): does NOT infer identity visually, does NOT assume engine
## indices are never recycled, does NOT format the EntityKey string (that is the
## M7 EntityKey Service). visible/hidden are DECOUPLED from spawn/destroy.
##
## Activation (KB Decision 24): inert unless explicitly driven — it has no
## _process/_physics_process; the exporter / a test calls scan() and
## update_visibility() each frame.
##
## Loaded via preload (no class_name).

const GROUP := &"structrender_entity"

var session_id: String = "session_dev"

## Test seam (S5b): (Node) -> int. Defaults to the recyclable engine index.
## Override to force a deterministic engine-index collision.
var engine_index_provider: Callable = func(n: Node) -> int: return n.get_instance_id()

## Test seam (S6b): when true, scan() is a no-op, so a node added to the entity
## group afterwards has no record and the exporter must abort loudly.
var frozen: bool = false

# engine_index -> record. record = {
#   instance_id:int, generation:int, node:Node, node_ref:WeakRef,
#   alive:bool, visible:bool, pixel_count:int, engine_class:String }
var _records: Dictionary = {}
var _next_instance_id: int = 1
var _events: Array[Dictionary] = []


func _engine_index(node: Node) -> int:
	return int(engine_index_provider.call(node))


## Detect spawns (new group member / reused index) and destroys (member gone).
## Call once per frame BEFORE rendering the entity-ID pass.
func scan(frame_id: int) -> void:
	if frozen:
		return
	var seen: Dictionary = {}
	for node in get_tree().get_nodes_in_group(GROUP):
		if not is_instance_valid(node):
			continue
		var ei := _engine_index(node)
		seen[ei] = node

		if not _records.has(ei):
			_spawn_record(ei, node, 0, frame_id)
			continue

		var rec: Dictionary = _records[ei]
		var same_node: bool = is_instance_valid(rec.node) and rec.node == node
		if not same_node:
			# The engine index now belongs to a different live node: reuse.
			# The old identity (if not already destroyed) is destroyed first.
			if rec.alive:
				_emit("destroy", rec, frame_id)
				rec.alive = false
			_spawn_record(ei, node, int(rec.generation) + 1, frame_id)

	# Destroys: previously-alive records whose node is gone / left the group.
	for ei in _records.keys():
		var rec: Dictionary = _records[ei]
		if rec.alive and not seen.has(ei):
			rec.alive = false
			rec.visible = false
			_emit("destroy", rec, frame_id)


## Derive visible/hidden from the entity-ID map. `counts` maps instance_id -> pixel_count.
## Call once per frame AFTER the entity-ID pass. Fully decoupled from spawn/destroy.
func update_visibility(counts: Dictionary, frame_id: int) -> void:
	for ei in _records.keys():
		var rec: Dictionary = _records[ei]
		if not rec.alive:
			continue
		var c: int = int(counts.get(rec.instance_id, 0))
		rec.pixel_count = c
		var vis := c > 0
		if vis and not rec.visible:
			rec.visible = true
			_emit("visible", rec, frame_id)
		elif not vis and rec.visible:
			rec.visible = false
			_emit("hidden", rec, frame_id)


## instance_id for a scene node, walking up to the nearest registered ancestor
## (a MeshInstance3D is usually a child of the entity body). Returns 0 if the
## node belongs to no registered entity. Aborts if a node IS in the entity group
## but has no record (should be impossible after scan()).
func instance_id_for(node: Node) -> int:
	var n := node
	while n != null:
		if n.is_in_group(GROUP):
			var ei := _engine_index(n)
			if not _records.has(ei) or not _records[ei].alive:
				push_error("EntityLifecycleAdapter: node '%s' is in group but unregistered" % n.name)
				return -1
			return int(_records[ei].instance_id)
		n = n.get_parent()
	return 0


## {instance_id -> {generation, engine_class, pixel_count, visible}} for alive entities.
func alive_entities() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	for ei in _records.keys():
		var rec: Dictionary = _records[ei]
		if not rec.alive:
			continue
		out.append({
			"instance_id": int(rec.instance_id),
			"generation": int(rec.generation),
			"engine_class": String(rec.engine_class),
			"pixel_count": int(rec.pixel_count),
			"visible": bool(rec.visible),
		})
	out.sort_custom(func(a, b): return a.instance_id < b.instance_id)
	return out


func events() -> Array[Dictionary]:
	return _events.duplicate(true)


func flush_events(path: String) -> void:
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_error("EntityLifecycleAdapter: cannot open %s (err %d)" % [path, FileAccess.get_open_error()])
		assert(false, "cannot open events file")
		return
	for ev in _events:
		f.store_line(JSON.stringify(ev))
	f.close()


func _spawn_record(ei: int, node: Node, generation: int, frame_id: int) -> void:
	var rec := {
		"instance_id": _next_instance_id,
		"generation": generation,
		"node": node,
		"alive": true,
		"visible": false,
		"pixel_count": 0,
		"engine_class": node.get_class(),
	}
	_next_instance_id += 1
	_records[ei] = rec
	_emit("spawn", rec, frame_id)


func _emit(event: String, rec: Dictionary, frame_id: int) -> void:
	_events.append({
		"session_id": session_id,
		"instance_id": int(rec.instance_id),
		"generation": int(rec.generation),
		"event": event,
		"frame_id": frame_id,
	})
