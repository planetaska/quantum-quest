extends TileMap

var GridSize = 8
var Dic = {}
var selectedTile

# Called when the node enters the scene tree for the first time.
func _ready():
	for x in GridSize:
		for y in GridSize:
			var loc = str(Vector2(x,y))
			Dic[loc] = {
				"Type": "Floor",
				"Position": loc
			}
			set_cell(0, Vector2(x,y), 0, Vector2i(0,0), 0)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	var pos = get_local_mouse_position()
	
	var tile = local_to_map(pos)
	
	for x in GridSize:
		for y in GridSize:
			set_cell(1, Vector2(x,y))
	
	if Dic.has(str(tile)):
		set_cell(1, tile, 1, Vector2i(0,0), 0)
		selectedTile = map_to_local(tile)
