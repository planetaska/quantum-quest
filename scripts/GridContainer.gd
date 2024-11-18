extends GridContainer

var cell = preload("res://components/grid-cell.tscn")

# Called when the node enters the scene tree for the first time.
func _ready():
	for i in range(64):
		add_child(cell.instantiate())
	
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
