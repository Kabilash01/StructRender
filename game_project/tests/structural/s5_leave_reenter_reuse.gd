extends SceneTree
## S5 — leave/re-enter identity stability + engine-index reuse
## (KB §11 Verification Scenario 5, KB §12 "Entity ID Reuse").
##
## (a) entity off-screen for several frames then back: id-map value identical
##     before/after, events show hidden -> visible (NO destroy/spawn), generation
##     unchanged.
## (b) free entity A, spawn B reusing A's engine index (test seam): B generation
##     > A, B instance_id != A, events show A destroy then B spawn.

const H := preload("res://tests/structural/structural_harness.gd")
const Adapter := preload("res://instrumentation/entity_lifecycle_adapter.gd")

const W_RES := 512
const H_RES := 288


func _initialize() -> void:
	await _part_a()
	await _part_b()
	H.finish(self)


func _part_a() -> void:
	var world := H.make_world(self)
	await process_frame
	H.add_prop(world, Vector3(0, 0, -12), Vector3(80, 80, 0.5), Color(0.3, 0.3, 0.3))
	var cube := H.add_entity(world, "Wanderer", Vector3(0, 0, -5), 1.5, Color(0.9, 0.7, 0.2))
	var home := cube.global_position
	var cam := H.make_camera(world, Vector3(0, 0, 0))
	await process_frame

	var adapter := H.make_adapter(self, "session_s5a")
	var ex := H.make_exporter(self, cam, world, adapter, "session_s5a")

	var id_before := -1
	var id_after := -1
	for f in 6:
		if f == 2:
			cube.global_position = Vector3(1000, 0, -5)   # off-screen, still in tree/group
		elif f == 4:
			cube.global_position = home
		await process_frame
		var fs: Dictionary = await ex.export_frame(f)
		H.check(not fs.is_empty(), "s5a frame %d exported" % f)
		var iid: int = adapter.instance_id_for(cube)
		var present := (fs.entity_id_map as PackedInt32Array).count(iid)
		if f == 1:
			id_before = iid
			H.check(present > 100, "s5a frame 1: entity visible in id map")
		elif f == 3:
			H.check(present == 0, "s5a frame 3: entity absent from id map (off-screen)")
		elif f == 5:
			id_after = iid
			H.check(present > 100, "s5a frame 5: entity visible again in id map")

	H.check(id_before == id_after and id_before > 0,
		"s5a: instance_id identical before(%d) and after(%d) leaving view" % [id_before, id_after])

	var ents: Array = adapter.alive_entities()
	H.check(ents.size() == 1 and ents[0].generation == 0,
		"s5a: generation unchanged (== 0) across leave/re-enter")

	var evs := _events_for(adapter, id_before)
	var seq: Array = []
	for e in evs:
		seq.append(e.event)
	H.check(not seq.has("destroy"), "s5a: NO destroy event on leaving view (got %s)" % str(seq))
	H.check(seq.count("spawn") == 1, "s5a: exactly one spawn (got %s)" % str(seq))
	H.check(seq.count("hidden") >= 1 and seq.count("visible") >= 2,
		"s5a: hidden then visible-again events present (got %s)" % str(seq))
	# order: spawn precedes first visible precedes hidden precedes second visible
	H.check(seq.find("hidden") > seq.find("visible") and seq.rfind("visible") > seq.find("hidden"),
		"s5a: event order is visible -> hidden -> visible (got %s)" % str(seq))

	adapter.flush_events("user://structural_debug/s5/events_a.jsonl")
	world.queue_free()
	adapter.queue_free()
	ex.queue_free()
	await process_frame


func _part_b() -> void:
	# Pure adapter logic with a deterministic engine-index seam.
	var adapter: Node = Adapter.new()
	root.add_child(adapter)
	adapter.session_id = "session_s5b"
	var idx_map := {}
	adapter.engine_index_provider = func(n: Node) -> int:
		return int(idx_map.get(n.get_instance_id(), n.get_instance_id()))

	var holder := Node3D.new()
	root.add_child(holder)

	var a := Node3D.new()
	a.name = "A"
	holder.add_child(a)
	a.add_to_group(&"structrender_entity")
	var a_seam_index := 999001
	idx_map[a.get_instance_id()] = a_seam_index
	adapter.scan(0)

	var a_ents: Array = adapter.alive_entities()
	H.check(a_ents.size() == 1, "s5b: A registered")
	var a_iid: int = a_ents[0].instance_id
	var a_gen: int = a_ents[0].generation
	H.check(a_gen == 0, "s5b: A generation == 0")

	# free A
	a.free()
	adapter.scan(1)   # A gone -> destroy
	H.check(adapter.alive_entities().is_empty(), "s5b: A destroyed after free")

	# spawn B, force it to reuse A's engine index
	var b := Node3D.new()
	b.name = "B"
	holder.add_child(b)
	b.add_to_group(&"structrender_entity")
	idx_map[b.get_instance_id()] = a_seam_index   # <-- reuse of A's (freed) engine index
	adapter.scan(2)

	var b_ents: Array = adapter.alive_entities()
	H.check(b_ents.size() == 1, "s5b: B registered")
	var b_iid: int = b_ents[0].instance_id
	var b_gen: int = b_ents[0].generation
	H.check(b_iid != a_iid, "s5b: B instance_id (%d) != A instance_id (%d)" % [b_iid, a_iid])
	H.check(b_gen > a_gen, "s5b: B generation (%d) > A generation (%d)" % [b_gen, a_gen])

	var seq: Array = []
	for e in adapter.events():
		seq.append("%s(iid=%d,gen=%d)" % [e.event, e.instance_id, e.generation])
	H.check(seq == [
			"spawn(iid=%d,gen=0)" % a_iid,
			"destroy(iid=%d,gen=0)" % a_iid,
			"spawn(iid=%d,gen=%d)" % [b_iid, b_gen],
		],
		"s5b: events = A spawn -> A destroy -> B spawn  (got %s)" % str(seq))

	adapter.flush_events("user://structural_debug/s5/events_b.jsonl")


func _events_for(adapter: Node, iid: int) -> Array:
	var out: Array = []
	for e in adapter.events():
		if e.instance_id == iid:
			out.append(e)
	return out
