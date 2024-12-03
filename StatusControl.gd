extends Control

enum States {
	BLOCK1,
	BLOCK2,
	BLOCK3,
	GOAL,
	READY,
	NOPATH,
	PLAY,
	SUCK,
	COMPLETE
}

#var states = ["block1", "block2", "block3", "goal", "play"]
var state = States.BLOCK1

var cone_locs = []

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func next_state():
	state += 1
	update_label()

func prev_state():
	state -= 1
	update_label()

func set_state(new_state):
	state = new_state
	update_label()

func update_label():
	var label = ""
	match state:
		States.BLOCK1:
			label = "Place Road Block 1"
		States.BLOCK2:
			label = "Place Road Block 2"
		States.BLOCK3:
			label = "Place Road Block 3"
		States.GOAL:
			label = "Place Goal"
		States.READY:
			label = "Ready!\nHit Start button."
		States.NOPATH:
			label = "No Path Found.\nTry Again."
		States.PLAY:
			label = "Searching..."
		States.SUCK:
			label = "Sucking Dirt..."
		States.COMPLETE:
			label = "Mission Complete!"
	
	$"../status_text".text = "[center]%s[/center]" % label
