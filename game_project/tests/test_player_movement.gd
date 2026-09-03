extends SceneTree
## Pressing a movement action actually moves the Player in the expected direction.

const TestHarness := preload("res://tests/harness.gd")

func _initialize() -> void:
	var scene := TestHarness.load_main_scene(self)
	var player: CharacterBody3D = scene.get_node("Player")

	await TestHarness.step_physics(self, 2)
	var start_z: float = player.global_position.z

	Input.action_press("move_forward")
	await TestHarness.step_physics(self, 20)
	Input.action_release("move_forward")

	var moved: float = start_z - player.global_position.z
	TestHarness.check(moved > 1.0, "move_forward moved Player -Z by >1m (moved %.2f)" % moved)

	var start_x: float = player.global_position.x
	Input.action_press("move_right")
	await TestHarness.step_physics(self, 20)
	Input.action_release("move_right")
	var moved_x: float = player.global_position.x - start_x
	TestHarness.check(moved_x > 1.0, "move_right moved Player +X by >1m (moved %.2f)" % moved_x)

	TestHarness.finish(self)
