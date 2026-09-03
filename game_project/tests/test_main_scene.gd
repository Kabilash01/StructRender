extends SceneTree
## The main scene instantiates and every entity node named in the spec is
## present with the exact contracted name (M1's EntityKey work depends on these).

const TestHarness := preload("res://tests/harness.gd")

func _initialize() -> void:
	var scene := TestHarness.load_main_scene(self)

	TestHarness.check(scene.has_node("Player"), "Player node present")
	TestHarness.check(scene.has_node("Player/MainCamera"), "MainCamera present under Player")
	TestHarness.check(scene.has_node("Arena"), "Arena node present")
	TestHarness.check(scene.has_node("Spawner"), "Spawner node present")
	for i in 5:
		TestHarness.check(scene.has_node("Coin_%d" % i), "Coin_%d present" % i)

	TestHarness.check(scene.get_node("Player") is CharacterBody3D, "Player is CharacterBody3D")
	TestHarness.check(scene.get_node("Player/MainCamera") is Camera3D, "MainCamera is Camera3D")
	TestHarness.check(scene.get_node("Coin_0") is Area3D, "Coin_0 is Area3D")

	# Spawner spawns Enemy_0 deferred on frame 1.
	await TestHarness.step_physics(self, 5)
	TestHarness.check(scene.has_node("Enemy_0"), "Enemy_0 spawned with contracted name")
	TestHarness.check(scene.get_node("Enemy_0") is CharacterBody3D, "Enemy_0 is CharacterBody3D")
	TestHarness.check_eq(root.get_node("GameState").enemy_alive, true, "GameState.enemy_alive true after spawn")

	TestHarness.finish(self)
