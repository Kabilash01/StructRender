extends Node
## Newline-delimited JSON logger (KB §8 "Logging & Observability").
## One JSON object per line, written to user://logs/session_<unix_ts>.jsonl and
## echoed to stdout. Fails loudly if the log file cannot be opened.

const LOG_DIR: String = "user://logs"

var _file: FileAccess
var _log_path: String = ""
var _start_usec: int = 0
var _game_end_logged: bool = false


func _ready() -> void:
	_start_usec = Time.get_ticks_usec()
	var err := DirAccess.make_dir_recursive_absolute(LOG_DIR)
	if err != OK and err != ERR_ALREADY_EXISTS:
		push_error("GameLog: cannotcreate log dir %s (err %d)" % [LOG_DIR, err])
		assert(false, "GameLog: cannotcreate log dir")
	var ts := int(Time.get_unix_time_from_system())
	_log_path = "%s/session_%d.jsonl" % [LOG_DIR, ts]
	_file = FileAccess.open(_log_path, FileAccess.WRITE)
	if _file == null:
		push_error("GameLog: cannotopen log file %s (err %d)" % [_log_path, FileAccess.get_open_error()])
		assert(false, "GameLog: cannotopen log file")

	GameState.coin_collected.connect(_on_coin_collected)
	GameState.enemy_spawned.connect(_on_enemy_spawned)
	GameState.enemy_destroyed.connect(_on_enemy_destroyed)
	GameState.game_won.connect(_on_game_won)

	log_event("game_start", {
		"total_coins": GameState.TOTAL_COINS,
		"headless": DisplayServer.get_name() == "headless",
	})


func _physics_process(_delta: float) -> void:
	if GameState.elapsed_ticks % 60 == 0:
		log_event("frame_tick", {
			"score": GameState.score,
			"coins_remaining": GameState.coins_remaining,
			"enemy_alive": GameState.enemy_alive,
		})


func _exit_tree() -> void:
	_log_game_end("exit_tree")


## Absolute path of the current session log file (used by tests).
func get_log_path() -> String:
	return _log_path


func log_event(event: String, extra: Dictionary = {}) -> void:
	var record := {
		"event": event,
		"tick": GameState.elapsed_ticks,
		"t_wall_s": float(Time.get_ticks_usec() - _start_usec) / 1_000_000.0,
	}
	record.merge(extra, true)
	var line := JSON.stringify(record)
	print(line)
	if _file != null:
		_file.store_line(line)
		_file.flush()


func _log_game_end(reason: String) -> void:
	if _game_end_logged:
		return
	_game_end_logged = true
	log_event("game_end", {
		"reason": reason,
		"won": GameState.won,
		"score": GameState.score,
	})
	if _file != null:
		_file.flush()


func _on_coin_collected(coin_name: String) -> void:
	log_event("coin_collected", {
		"name": coin_name,
		"score": GameState.score,
		"coins_remaining": GameState.coins_remaining,
	})


func _on_enemy_spawned(enemy_name: String) -> void:
	log_event("enemy_spawned", {"name": enemy_name})


func _on_enemy_destroyed(enemy_name: String) -> void:
	log_event("enemy_destroyed", {"name": enemy_name})


func _on_game_won() -> void:
	log_event("game_won", {
		"score": GameState.score,
		"ticks": GameState.elapsed_ticks,
	})
