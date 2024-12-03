extends CharacterBody2D


const SPEED = 300.0

var target_pos = Vector2(190, 130)


func _physics_process(delta):
	position += position.direction_to(target_pos) * SPEED * delta
	#var direction = global_position.direction_to(target_pos)
	#velocity = direction * SPEED
	#move_and_slide()
