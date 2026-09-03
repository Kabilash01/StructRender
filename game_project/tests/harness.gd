extends RefCounted
## Minimal dependency-free assertion helper for the M0 headless test scripts.
## Each test script is run standalone via:
##   godot --headless --path game_project --script res://tests/<name>.gd
## and exits 0 on success, non-zero on any failed check.
##
## Loaded with `preload("res://tests/harness.gd")` (no class_name, so it works
## without the editor having generated the global class cache).

static var _checks: int = 0
static var _failures: int = 0


static func check(cond: bool, msg: String) -> void:
	_checks += 1
	if cond:
		print("  ok   - ", msg)
	else:
		_failures += 1
		printerr("  FAIL - ", msg)


static func check_eq(actual: Variant, expected: Variant, msg: String) -> void:
	check(actual == expected, "%s (expected %s, got %s)" % [msg, expected, actual])


## Instantiate the main scene, register it as the current scene and disable the
## SimDriver so tests drive the world themselves. Returns the scene root.
static func load_main_scene(tree: SceneTree) -> Node:
	# Keep the headless SimDriver from also driving the world during a test.
	Engine.set_meta(&"structrender_no_sim", true)
	var sim := tree.root.get_node_or_null("SimDriver")
	if sim != null:
		sim.set_physics_process(false)
	var scene := (load("res://scenes/main.tscn") as PackedScene).instantiate()
	tree.root.add_child(scene)
	tree.current_scene = scene
	return scene


static func step_physics(tree: SceneTree, frames: int) -> void:
	for _i in frames:
		await tree.physics_frame


static func finish(tree: SceneTree) -> void:
	if _failures == 0:
		print("HARNESS_RESULT PASS (%d checks)" % _checks)
		tree.quit(0)
	else:
		printerr("HARNESS_RESULT FAIL (%d checks, %d failure(s))" % [_checks, _failures])
		tree.quit(1)
