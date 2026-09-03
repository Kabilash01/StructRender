extends SceneTree
## Project opens, autoloads are registered, and the input map has every action
## the spec requires.

const TestHarness := preload("res://tests/harness.gd")

func _initialize() -> void:
	TestHarness.check(root.has_node("GameState"), "GameState autoload present")
	TestHarness.check(root.has_node("GameLog"), "GameLog autoload present")
	TestHarness.check(root.has_node("SimDriver"), "SimDriver autoload present")

	for action in ["move_forward", "move_back", "move_left", "move_right", "interact"]:
		TestHarness.check(InputMap.has_action(action), "input action '%s' mapped" % action)

	TestHarness.check(ResourceLoader.exists("res://scenes/main.tscn"), "main.tscn resolvable")
	TestHarness.check(ResourceLoader.exists("res://scenes/enemy.tscn"), "enemy.tscn resolvable")

	var game_state := root.get_node("GameState")
	TestHarness.check_eq(game_state.coins_remaining, 5, "GameState starts with 5 coins")
	TestHarness.check_eq(game_state.won, false, "GameState starts not won")

	TestHarness.finish(self)
