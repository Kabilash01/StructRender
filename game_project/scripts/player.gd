extends CharacterBody3D
## Player capsule. Keyboard-only movement in the XZ plane at a fixed speed,
## gravity applied, collides with arena walls. No jump, no mouse-look
## (keeps headless runs deterministic).

@export var move_speed: float = 6.0

var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity", 9.8)


func _physics_process(delta: float) -> void:
	var input_dir := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	velocity.x = input_dir.x * move_speed
	velocity.z = input_dir.y * move_speed

	if is_on_floor():
		velocity.y = 0.0
	else:
		velocity.y -= _gravity * delta

	move_and_slide()
