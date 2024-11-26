import heapq
import json
import numpy as np

def print_board_state(board_state, start_pos, road_blocks, goals):
    """Prints a visual representation of the board state."""
    rows, cols = board_state.shape
    board_visual = [["." for _ in range(cols)] for _ in range(rows)]

    # Mark roadblocks
    for r, c in road_blocks:
        board_visual[r][c] = "R"

    # Mark goals
    for r, c in goals:
        board_visual[r][c] = "G"

    # Mark start position
    sr, sc = start_pos
    board_visual[sr][sc] = "S"

    # Print the board
    print("\n=== Board State ===")
    for row in board_visual:
        print(" ".join(row))
    print("===================")


def dijkstra_multiple_goals(board_state, start_pos, road_blocks, goals):
    rows, cols = board_state.shape
    visited = set()
    priority_queue = []

    # Add start position to the queue with distance 0
    heapq.heappush(priority_queue, (0, start_pos, frozenset()))  # (distance, position, collected_goals)

    # Parent map for reconstructing the path
    parent_map = {(start_pos, frozenset()): (None, None)}

    # Directions for moving on the board (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Convert goals to a set for fast lookup
    goal_set = set(goals)

    while priority_queue:
        current_distance, current_pos, collected_goals = heapq.heappop(priority_queue)

        # If the current state has already been visited, skip it
        if (current_pos, frozenset(collected_goals)) in visited:
            continue

        visited.add((current_pos, frozenset(collected_goals)))

        # If all goals are collected, reconstruct the path
        if collected_goals == goal_set:
            path = []
            state = (current_pos, frozenset(collected_goals))
            while state in parent_map:
                pos, prev_collected_goals = parent_map[state]
                if pos is not None:
                    path.append(list(pos))
                state = (pos, prev_collected_goals)
            return path[::-1]  # Return the path from start to goal

        # Explore neighbors
        for direction in directions:
            neighbor = (current_pos[0] + direction[0], current_pos[1] + direction[1])

            # Check if the neighbor is within bounds and not a roadblock
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in road_blocks:
                # Update the collected goals if the neighbor is a goal
                new_collected_goals = collected_goals | {neighbor} if neighbor in goals else collected_goals

                # Add neighbor to the priority queue with updated distance
                if (neighbor, frozenset(new_collected_goals)) not in visited:
                    heapq.heappush(priority_queue, (current_distance + 1, neighbor, new_collected_goals))
                    parent_map[(neighbor, frozenset(new_collected_goals))] = (current_pos, frozenset(collected_goals))

    # If no path is found, return an empty list
    return []



def main():
    # Load configuration from JSON
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 dijkstra.py <config_file>")
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
    print("=== Dijkstra's Algorithm Configuration ===")
    print("Board State Dimensions:", board_state.shape)
    print("Start Position:", start_pos)
    print("Road Blocks:", road_blocks)
    print("Goals:", goals)
    print("==========================================")

    # Print board
    print_board_state(board_state, start_pos, road_blocks, goals)

    # Run Dijkstra's algorithm
    route = dijkstra_multiple_goals(board_state, start_pos, road_blocks, goals)

    # Print the route
    if route:
        print("=== Shortest Path Found ===")
        print("Route (2D Array):", route)
        print("===========================")
    else:
        print("No path found to collect all goals.")


if __name__ == "__main__":
    main()
