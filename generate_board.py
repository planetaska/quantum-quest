import numpy as np
import json
from random import sample
from collections import deque


def generate_random_board_state(size=8, num_roadblocks=None, num_goals=1):
    if num_roadblocks + num_goals > size * size - 1:
        raise ValueError("Too many roadblocks and goals for the grid size.")

    # Generate all possible positions except (0, 0)
    all_positions = [(i, j) for i in range(size) for j in range(size) if (i, j) != (0, 0)]

    # Randomly select positions for roadblocks and goals
    roadblock_positions = set(sample(all_positions, num_roadblocks))
    remaining_positions = set(all_positions) - roadblock_positions
    goal_positions = set(sample(remaining_positions, num_goals))

    # Initialize board
    board_state = np.zeros((size, size), dtype=int)

    for block in roadblock_positions:
        board_state[block] = -1  # Use -1 to mark roadblocks

    for goal in goal_positions:
        board_state[goal] = 1  # Use 1 to mark goals

    # Check connectivity: Ensure (0, 0) isn't walled off and all goals are reachable
    if not are_goals_reachable(board_state, size):
        raise ValueError("(0, 0) is walled off or some goals are unreachable.")

    return board_state, list(roadblock_positions), list(goal_positions)


def are_goals_reachable(board_state, size):
    # Start position
    start = (0, 0)

    # Perform BFS to check reachability of all goals
    visited = set()
    queue = deque([start])
    goals_to_reach = set(zip(*np.where(board_state == 1)))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        if current in goals_to_reach:
            goals_to_reach.remove(current)

        # Explore neighbors
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                if neighbor not in visited and board_state[neighbor] != -1:
                    queue.append(neighbor)

    # Ensure (0, 0) can reach at least one open space and all goals
    return (len(goals_to_reach) == 0 and any(board_state[neighbor] == 0 for neighbor in get_neighbors(start, size)))


def get_neighbors(position, size):
    x, y = position
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    neighbors = [(x + dx, y + dy) for dx, dy in directions]
    return [n for n in neighbors if 0 <= n[0] < size and 0 <= n[1] < size]


def save_to_json(file_name, board_size, start_pos, road_blocks, goals):
    config = {
        "board_size": board_size,
        "start_pos": list(start_pos),
        "road_blocks": [list(block) for block in road_blocks],
        "goal_pos": [list(goal) for goal in goals]
    }
    with open(file_name, "w") as json_file:
        json.dump(config, json_file, indent=4)
    print(f"Configuration saved to {file_name}")


# Example usage
try:
    size = 8
    num_roadblocks = 10
    num_goals = 5

    board_state, roadblocks, goals = generate_random_board_state(size=size, num_roadblocks=num_roadblocks, num_goals=num_goals)

    print("Generated Board State:")
    print(board_state)
    print("Roadblocks:", roadblocks)
    print("Goals:", goals)

    # Save the configuration to a JSON file
    save_to_json("board_config.json", size, (0, 0), roadblocks, goals)

except ValueError as e:
    print("Error:", e)
