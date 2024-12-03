import numpy as np
import json
from collections import deque


def print_board_state(board_state, start_pos, road_blocks, goals, path=None):
    """Prints a visual representation of the board state."""
    rows, cols = board_state.shape
    board_visual = [["." for _ in range(cols)] for _ in range(rows)]

    # Mark roadblocks
    for r, c in road_blocks:
        board_visual[r][c] = "R"

    # Mark goals
    for r, c in goals:
        board_visual[r][c] = "G"

    # Mark path
    if path:
        for r, c in path:
            if (r, c) != start_pos and (r, c) not in goals and board_visual[r][c] == ".":
                board_visual[r][c] = "*"

    # Mark start position
    sr, sc = start_pos
    board_visual[sr][sc] = "S"

    # Print the board
    print("\n=== Board State ===")
    for row in board_visual:
        print(" ".join(row))
    print("===================")


def find_path(board_state, start, end, road_blocks_set):
    rows, cols = board_state.shape
    visited = set()
    queue = deque()
    queue.append((start, [start]))
    visited.add(start)

    while queue:
        current_pos, path = queue.popleft()
        if current_pos == end:
            return path
        r, c = current_pos
        # Check neighboring cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if (0 <= nr < rows and 0 <= nc < cols and
                neighbor not in visited and neighbor not in road_blocks_set):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    # No path found
    return None


def linear_search_collect_goals(board_state, start_pos, road_blocks, goals):
    rows, cols = board_state.shape
    road_blocks_set = set(road_blocks)
    goal_set = set(goals)
    collected_goals = set()
    path = [start_pos]
    current_pos = start_pos

    # Generate snake pattern coordinates, excluding roadblocks
    snake_coords = []
    for r in range(rows):
        row_coords = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in row_coords:
            pos = (r, c)
            if pos not in road_blocks_set:
                snake_coords.append(pos)

    # Remove positions before the start position in the snake pattern
    if start_pos in snake_coords:
        start_index = snake_coords.index(start_pos)
        snake_coords = snake_coords[start_index:]
    else:
        snake_coords.insert(0, start_pos)

    # Follow snake pattern, moving around roadblocks
    for next_pos in snake_coords[1:]:  # Exclude the start_pos itself
        # Find path from current_pos to next_pos
        sub_path = find_path(board_state, current_pos, next_pos, road_blocks_set)
        if sub_path is None:
            print(f"No path found from {current_pos} to {next_pos}")
            return []
        # Append sub_path to overall path, excluding the current position to avoid duplication
        path.extend(sub_path[1:])
        current_pos = next_pos
        if current_pos in goal_set:
            collected_goals.add(current_pos)
            if len(collected_goals) == len(goals):
                return path
    return path if len(collected_goals) == len(goals) else []


def main():
    # Load configuration from JSON
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 linear_search.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    try:
        with open(config_file, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found.")
        sys.exit(1)

    # Extract parameters
    board_size = config.get("board_size", 8)
    board_state = np.zeros((board_size, board_size))
    start_pos = tuple(config.get("start_pos", [0, 0]))
    road_blocks = [tuple(block) for block in config.get("road_blocks", [])]
    goals = [tuple(goal) for goal in config.get("goal_pos", [[board_size - 1, board_size - 1]])]

    # Print configuration
    print("=== Linear Search Configuration ===")
    print("Board State Dimensions:", board_state.shape)
    print("Start Position:", start_pos)
    print("Road Blocks:", road_blocks)
    print("Goals:", goals)

    # Run linear search
    route = linear_search_collect_goals(board_state, start_pos, road_blocks, goals)

    # Print board visualization
    print_board_state(board_state, start_pos, road_blocks, goals, route)

    # Print the route
    if route:
        print("=== Path Found ===")
        print("Route (2D Array):", route)
        # Print the route length
        print("Route Length:", len(route))
        print("==================")

        # Save route to JSON file
        with open("route.json", "w") as json_file:
            json.dump({"path": route}, json_file, indent=4)
        print("Path saved to route.json")

    else:
        print("No path found to collect all goals.")


if __name__ == "__main__":
    main()
