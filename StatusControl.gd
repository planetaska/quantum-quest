extends Control

enum States {
	BLOCK1,
	BLOCK2,
	BLOCK3,
	GOAL,
	PLAY
}

#var states = ["block1", "block2", "block3", "goal", "play"]
var state = States.BLOCK1

var cone_locs = []

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func next_state():
	state += 1

func prev_state():
	state -= 1
