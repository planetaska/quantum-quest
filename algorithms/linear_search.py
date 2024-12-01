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
            if (r, c) != start_pos and (r, c) not in goals:
                board_visual[r][c] = "*"

    # Mark start position
    sr, sc = start_pos
    board_visual[sr][sc] = "S"

    # Print the board
    print("\n=== Board State ===")
    for row in board_visual:
        print(" ".join(row))
    print("===================")


def linear_search_collect_goals(board_state, start_pos, road_blocks, goals):
    rows, cols = board_state.shape
    visited = set()
    path = []  # To store the path traversed
    current_pos = start_pos
    collected_goals = set()
    goal_set = set(goals)
    iterations = 0
    # Linear traversal across the grid
    for r in range(rows):
        for c in range(cols):
            iterations += 1
            pos = (r, c)

            # Skip roadblocks and already visited positions
            if pos in road_blocks or pos in visited:
                continue

            # Mark as visited
            visited.add(pos)

            # Append to path
            path.append(list(pos))

            # Check if the position is a goal
            if pos in goal_set:
                collected_goals.add(pos)

                # Stop if all goals are collected
                if collected_goals == goal_set:
                    print(f"Number of iterations required to find path: {iterations}")
                    return path

    # If the loop completes without collecting all goals, return an empty list
    print("Not all goals could be collected.")
    return []


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
