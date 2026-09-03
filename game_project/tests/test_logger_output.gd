extends SceneTree
## The Logger writes newline-delimited JSON: every line parses, and the required
## events appear.

const TestHarness := preload("res://tests/harness.gd")

func _initialize() -> void:
	var scene := TestHarness.load_main_scene(self)
	var game_state := root.get_node("GameState")
	var game_log := root.get_node("GameLog")

	await TestHarness.step_physics(self, 5)
	game_state.collect_coin("Coin_0")
	game_state.destroy_enemy("Enemy_0")
	await TestHarness.step_physics(self, 65)  # cover at least one frame_tick sample

	var path: String = game_log.get_log_path()
	TestHarness.check(path != "" and FileAccess.file_exists(path), "log file exists at %s" % path)

	var text := FileAccess.get_file_as_string(path)
	var lines := text.split("\n", false)
	TestHarness.check(lines.size() > 0, "log file has content")

	var events := {}
	var all_valid := true
	for line in lines:
		var parsed: Variant = JSON.parse_string(line)
		if typeof(parsed) != TYPE_DICTIONARY:
			all_valid = false
			printerr("  bad JSON line: ", line)
			continue
		if not (parsed.has("event") and parsed.has("tick") and parsed.has("t_wall_s")):
			all_valid = false
			printerr("  line missing required keys: ", line)
			continue
		events[parsed["event"]] = true

	TestHarness.check(all_valid, "every log line is a valid JSON object with event/tick/t_wall_s")
	for required in ["game_start", "coin_collected", "enemy_destroyed", "frame_tick"]:
		TestHarness.check(events.has(required), "log contains '%s' event" % required)

	TestHarness.finish(self)
