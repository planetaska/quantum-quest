extends Control

var states = ["placing_goal", "placing_blocks", "play"]
var state = 0

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func next_state():
	state += 1

func prev_state():
	state -= 1
