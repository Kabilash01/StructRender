extends SceneTree
## The win condition (all coins collected AND enemy destroyed) fires `game_won`
## exactly once.

const TestHarness := preload("res://tests/harness.gd")

var _won_count: int = 0

func _initialize() -> void:
	var scene := TestHarness.load_main_scene(self)
	var game_state := root.get_node("GameState")
	game_state.game_won.connect(_on_game_won)

	await TestHarness.step_physics(self, 5)  # let the spawner register the enemy

	game_state.collect_coin("Coin_0")
	game_state.collect_coin("Coin_1")
	game_state.collect_coin("Coin_2")
	game_state.collect_coin("Coin_3")
	TestHarness.check_eq(game_state.won, false, "not won with coins left")

	game_state.collect_coin("Coin_4")
	TestHarness.check_eq(game_state.won, false, "not won while enemy alive")

	game_state.destroy_enemy("Enemy_0")
	TestHarness.check_eq(game_state.won, true, "won after last coin + enemy destroyed")

	# Redundant calls must not re-emit game_won.
	game_state.destroy_enemy("Enemy_0")
	game_state.collect_coin("phantom")

	await TestHarness.step_physics(self, 2)
	TestHarness.check_eq(_won_count, 1, "game_won emitted exactly once")

	TestHarness.finish(self)


func _on_game_won() -> void:
	_won_count += 1
