extends Node
## Authoritative gameplay state for the M0 "Arena Collector" game.
## Godot is the sole source of gameplay truth (KB Decision 2) — nothing else
## may mutate these fields.

signal coin_collected(coin_name: String)
signal enemy_spawned(enemy_name: String)
signal enemy_destroyed(enemy_name: String)
signal game_won()

const TOTAL_COINS: int = 5

var score: int = 0
var coins_remaining: int = TOTAL_COINS
var enemy_alive: bool = false
var elapsed_ticks: int = 0
var won: bool = false


func _physics_process(_delta: float) -> void:
	elapsed_ticks += 1


func register_enemy_spawned(enemy_name: String) -> void:
	enemy_alive = true
	enemy_spawned.emit(enemy_name)


func collect_coin(coin_name: String) -> void:
	score += 1
	coins_remaining -= 1
	coin_collected.emit(coin_name)
	_check_win()


func destroy_enemy(enemy_name: String) -> void:
	enemy_alive = false
	enemy_destroyed.emit(enemy_name)
	_check_win()


func _check_win() -> void:
	if won:
		return
	if coins_remaining == 0 and not enemy_alive:
		won = true
		game_won.emit()
