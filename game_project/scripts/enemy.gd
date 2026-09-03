extends CharacterBody3D
## Enemy_0. Pure pursuit: moves toward the Player at a fixed speed each physics
## tick (no pathfinding). Destroyed when the Player holds `interact` within
## INTERACT_RANGE metres.

@export var move_speed: float = 3.0

const INTERACT_RANGE: float = 2.0

var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity", 9.8)
var _player: Node3D


func _ready() -> void:
	_player = get_tree().current_scene.get_node_or_null("Player")


func _physics_process(delta: float) -> void:
	if _player == null:
		return

	var to_player := _player.global_position - global_position
	to_player.y = 0.0
	var dist := to_player.length()

	if dist > 0.001:
		var dir := to_player / dist
		velocity.x = dir.x * move_speed
		velocity.z = dir.z * move_speed
	else:
		velocity.x = 0.0
		velocity.z = 0.0

	if is_on_floor():
		velocity.y = 0.0
	else:
		velocity.y -= _gravity * delta

	move_and_slide()

	if dist <= INTERACT_RANGE and Input.is_action_pressed("interact"):
		GameState.destroy_enemy(name)
		queue_free()
