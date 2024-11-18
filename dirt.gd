extends CharacterBody2D

const SPEED = 300.0

var adding = false

func _unhandled_input(event):
	if event.is_action_pressed("LeftClick"):
		adding = true

func _input(event):
	if event.is_action_released("LeftClick"):
		adding = false

func _physics_process(delta):
	if adding:
		#print(get_tree().root.get_node("main-screen/tile_map/TileMap").selectedTile)
		#print($"../tile_map/TileMap".selectedTile)
		global_position = Vector2($"../tile_map/TileMap".selectedTile) + Vector2(60,60)
