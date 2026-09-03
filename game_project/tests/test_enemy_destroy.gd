extends SceneTree
## The spawner produces exactly one enemy, and the Player can destroy it with
## `interact` in range.

const TestHarness := preload("res://tests/harness.gd")

func _initialize() -> void:
	var scene := TestHarness.load_main_scene(self)
	var game_state := root.get_node("GameState")

	await TestHarness.step_physics(self, 5)
	TestHarness.check(scene.has_node("Enemy_0"), "Enemy_0 spawned")
	TestHarness.check_eq(_count_enemies(scene), 1, "exactly one enemy spawned")

	var player: CharacterBody3D = scene.get_node("Player")
	var enemy: CharacterBody3D = scene.get_node("Enemy_0")
	player.global_position = enemy.global_position + Vector3(1.0, 0.0, 0.0)

	Input.action_press("interact")
	await TestHarness.step_physics(self, 5)
	Input.action_release("interact")

	TestHarness.check_eq(game_state.enemy_alive, false, "GameState.enemy_alive false after destroy")
	TestHarness.check(not scene.has_node("Enemy_0"), "Enemy_0 freed itself after destroy")

	TestHarness.finish(self)


func _count_enemies(scene: Node) -> int:
	var n := 0
	for child in scene.get_children():
		if child.name.begins_with("Enemy_"):
			n += 1
	return n
