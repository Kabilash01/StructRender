extends Node
## Headless simulation driver. Active ONLY under `--headless` (DisplayServer
## "headless") or when the `structrender_sim` feature tag is present.
##
## Drives a deterministic closed-loop input sequence: it steers the Player with
## the same movement actions a human would use, toward each fixed coin position
## in order, then toward Enemy_0, pressing `interact` in range. No randomness —
## the fixed scene layout makes every run identical. On win it asserts
## GameState.won and quits 0; on exceeding the tick budget it logs an error and
## quits 1.
##
## A headless test that drives the world itself sets the Engine meta
## `structrender_no_sim` so this driver stays out of the way.

const MAX_TICKS: int = 3000
const DISABLE_META: StringName = &"structrender_no_sim"
const ARRIVE_DEADZONE: float = 0.15
const INTERACT_DISTANCE: float = 1.8

var _active: bool = false
var _player: CharacterBody3D
var _move_actions: PackedStringArray = [
	"move_forward", "move_back", "move_left", "move_right", "interact",
]


func _ready() -> void:
	_active = (DisplayServer.get_name() == "headless" or OS.has_feature("structrender_sim")) \
		and not Engine.has_meta(DISABLE_META)
	if not _active:
		set_physics_process(false)
		return
	GameLog.log_event("sim_driver_start", {"max_ticks": MAX_TICKS})


func _physics_process(_delta: float) -> void:
	if not _active or Engine.has_meta(DISABLE_META):
		return

	if GameState.won:
		_release_all()
		assert(GameState.coins_remaining == 0 and not GameState.enemy_alive,
			"SimDriver: game_won with inconsistent GameState")
		GameLog.log_event("sim_driver_done", {"ticks": GameState.elapsed_ticks})
		get_tree().quit(0)
		return

	if GameState.elapsed_ticks > MAX_TICKS:
		_release_all()
		GameLog.log_event("error", {
			"where": "sim_driver",
			"message": "win state not reached within tick budget",
			"coins_remaining": GameState.coins_remaining,
			"enemy_alive": GameState.enemy_alive,
		})
		get_tree().quit(1)
		return

	if _player == null:
		_player = _get_scene_node("Player") as CharacterBody3D
		if _player == null:
			return

	var target := _current_target()
	if target == null:
		_release_all()
		return

	var to_target: Vector3 = (target as Node3D).global_position - _player.global_position
	to_target.y = 0.0

	_drive_axis("move_right", "move_left", to_target.x)
	_drive_axis("move_back", "move_forward", to_target.z)

	if (target as Node3D).name == "Enemy_0" and to_target.length() <= INTERACT_DISTANCE:
		Input.action_press("interact")
	else:
		Input.action_release("interact")


## Lowest-index coin still in the tree, else Enemy_0, else null.
func _current_target() -> Node:
	for i in GameState.TOTAL_COINS:
		var coin := _get_scene_node("Coin_%d" % i)
		if coin != null and is_instance_valid(coin):
			return coin
	var enemy := _get_scene_node("Enemy_0")
	if enemy != null and is_instance_valid(enemy):
		return enemy
	return null


func _drive_axis(pos_action: String, neg_action: String, value: float) -> void:
	if value > ARRIVE_DEADZONE:
		Input.action_press(pos_action)
		Input.action_release(neg_action)
	elif value < -ARRIVE_DEADZONE:
		Input.action_press(neg_action)
		Input.action_release(pos_action)
	else:
		Input.action_release(pos_action)
		Input.action_release(neg_action)


func _release_all() -> void:
	for action in _move_actions:
		Input.action_release(action)


func _get_scene_node(node_name: String) -> Node:
	var scene := get_tree().current_scene
	if scene == null:
		return null
	return scene.get_node_or_null(node_name)
