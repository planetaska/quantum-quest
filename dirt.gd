extends CharacterBody2D

var adding = false

func saveConfig(x, y):
	# user://
	# Windows: %APPDATA%\Godot\app_userdata\[project_name]
	# macOS: ~/Library/Application Support/Godot/app_userdata/[project_name]
	# Linux: ~/.local/share/godot/app_userdata/[project_name]
	
	var save_file = FileAccess.open("user://config.json", FileAccess.WRITE)
	#print($"../StatusControl".cone_locs)
	var road_blocks = $"../StatusControl".cone_locs
	var config = {
		"board_size": 8,
		"start_pos": [0, 0],
		"road_blocks": road_blocks,
		"goal_pos": [[x, y]]
	}
	var json_string = JSON.stringify(config)
	#print(json_string)
	save_file.store_line(json_string)

func _unhandled_input(event):
	if event.is_action_pressed("LeftClick"):
		if ($"../StatusControl".state == $"../StatusControl".States.GOAL):
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
		
		$"../StatusControl".next_state()
		adding = false
