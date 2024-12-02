extends CharacterBody2D

const SPEED = 300.0

var adding = false

func saveConfig(x, y):
	var save_file = FileAccess.open("./algorithms/config.json", FileAccess.WRITE)
	var config = {
		"board_size": 8,
		"start_pos": [0, 0],
		"road_blocks": [[1,1],[2,2],[3, 3], [4, 4]],
		"goal_pos": [[x, y]]
	}
	var json_string = JSON.stringify(config)
	save_file.store_line(json_string)

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
		var selectedTile = $"../tile_map/TileMap".selectedTile
		#print(selectedTile)
		global_position = Vector2(selectedTile) + Vector2(160,100)
		
		var x = (int(selectedTile[0]) -28) / 56
		var y = (int(selectedTile[1]) -28) / 56
		saveConfig(x,y)
		adding = false
