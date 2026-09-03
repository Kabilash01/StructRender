extends SceneTree
## M1 contract: the Arena scene's entities are in the `structrender_entity`
## group (and non-entities are not), and the EntityLifecycleAdapter assigns
## stable instance_ids + spawn events for them. Headless-safe (no rendering) —
## runs in the M0 suite. GPU structural export is covered by S1-S6
## (scripts/test_structural.sh).

const TestHarness := preload("res://tests/harness.gd")
const Adapter := preload("res://instrumentation/entity_lifecycle_adapter.gd")

const GROUP := &"structrender_entity"


func _initialize() -> void:
	var scene := TestHarness.load_main_scene(self)
	await TestHarness.step_physics(self, 5)  # let Spawner add Enemy_0

	for ename in ["Player", "Coin_0", "Coin_1", "Coin_2", "Coin_3", "Coin_4", "Enemy_0"]:
		var n := scene.get_node_or_null(ename)
		TestHarness.check(n != null and n.is_in_group(GROUP), "%s is in structrender_entity group" % ename)

	for non_ent in ["Arena", "Spawner"]:
		TestHarness.check(not scene.get_node(non_ent).is_in_group(GROUP), "%s is NOT an entity" % non_ent)
	TestHarness.check(not scene.get_node("Player/MainCamera").is_in_group(GROUP), "MainCamera is NOT an entity")

	var adapter := Adapter.new()
	root.add_child(adapter)
	adapter.scan(0)
	var ents := adapter.alive_entities()
	TestHarness.check_eq(ents.size(), 7, "adapter registered all 7 entities")

	var ids := {}
	for e in ents:
		ids[e.instance_id] = true
		TestHarness.check_eq(e.generation, 0, "entity id %d generation == 0" % e.instance_id)
	TestHarness.check_eq(ids.size(), 7, "7 distinct instance_ids")
	TestHarness.check(not ids.has(0), "no entity was assigned instance_id 0 (0 == background)")

	var spawn_events := 0
	for ev in adapter.events():
		if ev.event == "spawn":
			spawn_events += 1
	TestHarness.check_eq(spawn_events, 7, "one spawn event per entity")

	TestHarness.finish(self)
