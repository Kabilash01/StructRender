extends Node3D
## On frame 1 spawns exactly one enemy (Enemy_0) from enemy.tscn as a child of
## the main scene, and registers the spawn with GameState.

const ENEMY_SCENE: PackedScene = preload("res://scenes/enemy.tscn")
const SPAWN_POSITION: Vector3 = Vector3(0.0, 1.0, -8.0)

var _spawned: bool = false


func _ready() -> void:
	# Deferred: the main scene is still setting up children during _ready.
	_spawn.call_deferred()


func _spawn() -> void:
	if _spawned:
		return
	_spawned = true
	var enemy: CharacterBody3D = ENEMY_SCENE.instantiate()
	enemy.name = "Enemy_0"
	enemy.add_to_group(&"structrender_entity")  # M1: structural-export entity (see data_pipeline/schemas/frame_structure.md §5)
	get_tree().current_scene.add_child(enemy)
	enemy.global_position = SPAWN_POSITION
	GameState.register_enemy_spawned(enemy.name)
