extends Button

@onready var timer = $"../Timer"

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func _on_pressed():
	var node = $"../robot"
	#Global.model = "cgrover3"
	
	var py_path = "/Users/aska/lab/python/qiskit/bin/python3"
	var alg_path = ["./algorithms/%s.py" % Global.model, "%s/config.json" % OS.get_user_data_dir()]
	#print(alg_path)
	#return
	var output = []
	
	OS.execute(py_path, alg_path, output)
	
	print(output)
	
	if "No path found" in output[0]:
		$"../StatusControl".set_state($"../StatusControl".States.NOPATH)
		return
	
	#return
	
	#var route = [[0, 0], [0, 2], [2, 1], [5, 5], [2, 1], [2, 5], [6, 6], [5, 1], [4, 5], [0, 7], [3, 7], [4, 6], [1, 7], [2, 0], [2, 7], [3, 2], [2, 4], [4, 7], [0, 4], [2, 3], [0, 6], [1, 3], [3, 7], [3, 4], [7, 0], [1, 3], [6, 3], [2, 1], [0, 1], [2, 3], [5, 1], [5, 7], [2, 1], [6, 7], [3, 2], [7, 4], [2, 5], [0, 1], [6, 1], [6, 5], [4, 7], [2, 2], [3, 5], [7, 2], [5, 0], [7, 5], [0, 5], [7, 4], [3, 7], [7, 7]]
	var file = "./route.json"
	var json_as_text = FileAccess.get_file_as_string(file)
	var route = JSON.parse_string(json_as_text)
	
	if route:
		print(route.path)
		$"../StatusControl".set_state($"../StatusControl".States.PLAY)
		Global.score = route.path.size()
		
		for pos in route.path:
			await get_tree().create_timer(0.4).timeout
			#node.position = Vector2( (pos[0]*56)+190, (pos[1]*56)+130 )
			node.target_pos = Vector2( (pos[0]*56)+190, (pos[1]*56)+130 )
		
		await get_tree().create_timer(0.4).timeout
		$"../StatusControl".set_state($"../StatusControl".States.SUCK)
		await get_tree().create_timer(0.4).timeout
		$"../dirt".call_deferred("queue_free")
		
		$"../StatusControl".set_state($"../StatusControl".States.COMPLETE)
		await get_tree().create_timer(1).timeout
		get_tree().change_scene_to_file("res://result-screen.tscn")
