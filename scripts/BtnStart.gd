extends Button

@onready var timer = $"../Timer"

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func _on_pressed():
	var node = $"../robot"
	
	var route = [[0, 0], [0, 2], [2, 1], [5, 5], [2, 1], [2, 5], [6, 6], [5, 1], [4, 5], [0, 7], [3, 7], [4, 6], [1, 7], [2, 0], [2, 7], [3, 2], [2, 4], [4, 7], [0, 4], [2, 3], [0, 6], [1, 3], [3, 7], [3, 4], [7, 0], [1, 3], [6, 3], [2, 1], [0, 1], [2, 3], [5, 1], [5, 7], [2, 1], [6, 7], [3, 2], [7, 4], [2, 5], [0, 1], [6, 1], [6, 5], [4, 7], [2, 2], [3, 5], [7, 2], [5, 0], [7, 5], [0, 5], [7, 4], [3, 7], [7, 7]]
	
	for pos in route:
		await get_tree().create_timer(0.3).timeout
		node.position = Vector2( (pos[0]*64)+90, (pos[1]*64)+90 )