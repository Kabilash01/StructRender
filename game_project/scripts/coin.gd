extends Area3D
## A collectible coin. When the Player enters its area it reports the collection
## to GameState (the source of truth) and then frees itself.

func _ready() -> void:
	body_entered.connect(_on_body_entered)


func _on_body_entered(body: Node3D) -> void:
	if body.name != "Player":
		return
	GameState.collect_coin(name)
	queue_free()
