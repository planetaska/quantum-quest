extends CharacterBody2D

var adding = false

func _unhandled_input(event):
	if event.is_action_pressed("LeftClick"):
		if ($"../StatusControl".state == $"../StatusControl".States.BLOCK2):
			adding = true

func _input(event):
	if event.is_action_released("LeftClick"):
		adding = false

func _physics_process(delta):
	if adding:
		#print(get_tree().root.get_node("main-screen/tile_map/TileMap").selectedTile)
		#print($"../tile_map/TileMap".selectedTile)
		var selectedTile = $"../tile_map/TileMap".selectedTile
		#print(selectedTile)
		global_position = Vector2(selectedTile) + Vector2(160,100)
		
		var x = (int(selectedTile[0]) -28) / 56
		var y = (int(selectedTile[1]) -28) / 56
		#saveConfig(x,y)
		
		$"../StatusControl".cone_locs.append([x, y])
		$"../StatusControl".next_state()
		adding = false
