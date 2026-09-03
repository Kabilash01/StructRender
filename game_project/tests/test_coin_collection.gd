extends SceneTree
## Entering a coin's area increments the score and frees the coin.

const TestHarness := preload("res://tests/harness.gd")

func _initialize() -> void:
	var scene := TestHarness.load_main_scene(self)
	var game_state := root.get_node("GameState")
	var player: CharacterBody3D = scene.get_node("Player")
	var coin: Area3D = scene.get_node("Coin_0")

	await TestHarness.step_physics(self, 2)
	TestHarness.check_eq(game_state.score, 0, "score starts at 0")

	player.global_position = coin.global_position
	await TestHarness.step_physics(self, 5)

	TestHarness.check_eq(game_state.score, 1, "score is 1 after collecting Coin_0")
	TestHarness.check_eq(game_state.coins_remaining, 4, "coins_remaining is 4")
	TestHarness.check(not scene.has_node("Coin_0"), "Coin_0 freed itself after collection")

	TestHarness.finish(self)
