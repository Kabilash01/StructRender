## Record the PLAYABLE game (main.tscn) for training — KB Decision 87.
##
## The ladder was trained on a scene the recorder builds in code; the playable
## arena collector (player, coins, spawned enemies) was never seen. This records
## real gameplay with the SAME exporter/recorder/adapter the dataset uses, so the
## output is the same schema and passes the same M2 validator.
##
## Nothing in the game is modified:
##  * the game's own SimDriver is kept out via the `structrender_no_sim` meta it
##    already honours; the player is steered by pressing the game's own input
##    actions, exactly as a human would;
##  * each recorded frame advances the game by EXACTLY ONE physics step
##    (max_physics_steps_per_frame = 1, unpause -> physics step -> pause), and the
##    tree stays paused while the exporter renders its passes, so depth, normal,
##    albedo, ID and native RGB all show the same instant. Without the pause the
##    player and enemies move between passes and the buffers disagree.
## Diversity per session comes from STRUCTRENDER_SESSION_VARIANT: start position
## and a seeded wandering route (random arena points and the nearest coin).
extends SceneTree

const H := preload("res://tests/structural/structural_harness.gd")
const DH := preload("res://tests/dataset/dataset_harness.gd")
const FRAMES_DEFAULT := 990
const ARENA := 8.0

var _held: Array = []

func _steer(dir: Vector2) -> void:
	for a in _held:
		Input.action_release(a)
	_held.clear()
	if dir.length() < 0.05:
		return
	dir = dir.normalized()
	var pairs := [["move_right", dir.x], ["move_left", -dir.x], ["move_back", dir.y], ["move_forward", -dir.y]]
	for p in pairs:
		if p[1] > 0.0:
			Input.action_press(p[0], p[1]); _held.append(p[0])

func _initialize() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	Engine.max_physics_steps_per_frame = 1
	Engine.set_meta(&"structrender_no_sim", true)
	var frames := int(OS.get_environment("SR_FRAMES")) if OS.has_environment("SR_FRAMES") else FRAMES_DEFAULT
	var variant := int(OS.get_environment("STRUCTRENDER_SESSION_VARIANT"))
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260928 + variant * 1000

	var main: Node3D = load("res://scenes/main.tscn").instantiate()
	root.add_child(main)
	# The game's spawner adds enemies to get_tree().current_scene. Loaded by
	# script rather than as the main scene, that is null unless set — the first
	# test run recorded a game in which no enemy ever spawned.
	current_scene = main
	await process_frame
	var player: CharacterBody3D = main.get_node("Player")
	var cam: Camera3D = player.get_node("MainCamera")
	player.global_position = Vector3(rng.randf_range(-6.0, 6.0), 1.0, rng.randf_range(-6.0, 6.0))
	await physics_frame
	await process_frame

	var sid := DH.new_session_id()
	var adapter := H.make_adapter(self, sid)
	var ex := H.make_exporter(self, cam, main, adapter, sid)
	var rec := DH.make_recorder(self, sid, DH.dataset_root(), ex, adapter, cam)
	for n in [adapter, ex, rec]:
		n.process_mode = Node.PROCESS_MODE_ALWAYS
	if not rec.begin():
		printerr("GAME_RECORD FAIL recorder.begin refused"); quit(1); return
	paused = true

	var target := Vector2(player.global_position.x, player.global_position.z)
	var hold := 0
	var tick := 0
	for i in frames:
		var here := Vector2(player.global_position.x, player.global_position.z)
		if hold <= 0 or here.distance_to(target) < 0.6:
			var coins := main.get_tree().get_nodes_in_group(&"structrender_entity").filter(
				func(n): return n.name.begins_with("Coin") and n.is_inside_tree())
			if coins.size() > 0 and rng.randf() < 0.5:
				var c: Node3D = coins[rng.randi_range(0, coins.size() - 1)]
				target = Vector2(c.global_position.x, c.global_position.z)
			else:
				target = Vector2(rng.randf_range(-ARENA, ARENA), rng.randf_range(-ARENA, ARENA))
			hold = rng.randi_range(60, 180)
		hold -= 1
		_steer(target - here)
		paused = false
		await physics_frame        # the one physics step this frame runs...
		await process_frame        # ...has completed; nothing else moves until unpause
		paused = true
		tick += 1
		var r: Dictionary = await rec.record_frame(tick)
		if r.is_empty():
			printerr("GAME_RECORD FAIL frame %d: %s" % [i, rec.get("last_error")]); quit(1); return
	_steer(Vector2.ZERO)
	rec.add_temporal_sequence("gameplay_0", 0, frames - 1)
	rec.finish()
	printerr("GAME_RECORD OK session=%s frames=%d variant=%d" % [sid, frames, variant])
	quit(0)
