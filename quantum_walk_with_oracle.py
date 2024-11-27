import sys
import json
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def index_to_binary(index, num_qubits):
    """Convert grid index to binary representation."""
    return format(index, f'0{num_qubits}b')


def binary_to_index(binary_str):
    """Convert binary string to an integer index."""
    return int(binary_str, 2)


def index_to_coordinates(index, grid_size):
    """Convert a flattened index to grid coordinates within bounds."""
    rows, cols = grid_size, grid_size
    return index // cols, index % cols

def flatten_coordinates(row, col, grid_size):
    """Convert (row, col) grid coordinates to a flattened index."""
    return row * grid_size + col

def enforce_valid_moves(route, goals):
    """Ensure the route consists of valid moves while including reachable goals."""
    print("Raw Route Before Filtering:", route)  # Debugging line
    if not route:
        return []

    valid_route = [route[0]]  # Start with the initial position
    remaining_goals = set(goals)  # Track remaining goals

    for pos in route:
        # If this position is one of the goals, add it
        if pos in remaining_goals:
            valid_route.append(pos)
            remaining_goals.remove(pos)

        # Ensure the move is adjacent (optional for direct connectivity)
        elif abs(valid_route[-1][0] - pos[0]) + abs(valid_route[-1][1] - pos[1]) == 1:
            valid_route.append(pos)

    # Add any remaining goals that were not part of the route
    for goal in remaining_goals:
        valid_route.append(goal)

    print("Filtered Valid Route:", valid_route)  # Debugging line
    return valid_route




def add_oracle(qc, goals, num_qubits):
    """Add an oracle that marks the goal states."""
    grid_size = int(np.sqrt(2 ** (num_qubits - 1)))
    flattened_goals = [flatten_coordinates(goal[0], goal[1], grid_size) for goal in goals]
#    print("Flattened Goals:", flattened_goals)  # Debugging line
    for goal in flattened_goals:
        binary_goal = index_to_binary(goal, num_qubits - 1)
 #       print(f"Marking goal at index {goal} (binary: {binary_goal})")  # Debugging line
        for i, bit in enumerate(binary_goal):
            if bit == '0':
                qc.x(i)  # Flip qubits for 0s in the binary representation
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)  # Multi-controlled X gate
        for i, bit in enumerate(binary_goal):
            if bit == '0':
                qc.x(i)  # Unflip qubits




def add_coin_operator(qc, num_qubits):
    """Add a coin operator to distribute amplitude equally among directions."""
    qc.h(range(num_qubits - 1))  # Apply Hadamard gates to all "position" qubits


def add_shift_operator(qc, num_qubits):
    """Add a shift operator to move the walker based on the coin."""
    for i in range(num_qubits - 1):
        qc.cx(i, num_qubits - 1)  # Move based on the state of the coin qubit


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


def quantum_walk(board_state, start_pos, road_blocks, goals, use_simulator=True, iterations=3):
    grid_size = board_state.shape[0]
    num_qubits = int(np.ceil(np.log2(grid_size ** 2))) + 1  # +1 for auxiliary qubit
    qc = QuantumCircuit(num_qubits)

    # Initialize in equal superposition
    qc.h(range(num_qubits - 1))

    # Perform quantum walk iterations
    for _ in range(iterations):
        # Apply the oracle
        add_oracle(qc, goals, num_qubits)

        # Apply the coin operator
        add_coin_operator(qc, num_qubits)

        # Apply the shift operator
        add_shift_operator(qc, num_qubits)

    # Simulate measurement
    if use_simulator:
        simulator = AerSimulator()
        qc.measure_all()
        transpiled_qc = transpile(qc, simulator)
        result = simulator.run(transpiled_qc, shots=1024).result()
        counts = result.get_counts()

        # Convert measurement results to grid positions
        route = []
        for state, freq in counts.items():
            index = binary_to_index(state.split(" ")[0])  # Get the index from the binary string
            if freq > 0:
                coord = index_to_coordinates(index, grid_size)
                if 0 <= coord[0] < grid_size and 0 <= coord[1] < grid_size:  # Validate bounds
                    route.append(coord)

        # Deduplicate and sort the route
        route = sorted(set(route))

        # Enforce valid moves
        route = enforce_valid_moves(route, goals)

        # Plot measurement histogram
        plt.figure(figsize=(8, 6))
        plot_histogram(counts)
        plt.show()

        print("Route (2D Array):", route)
        return route

    else:
        # If no simulator is used, just return an empty route
        return []


def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.json"

    # Load configuration from file
    with open(config_file, "r") as file:
        config = json.load(file)

    # Create board state dynamically based on size
    board_size = config["board_size"]
    board_state = np.zeros((board_size, board_size))

    # Extract other parameters
    start_pos = tuple(config["start_pos"])
    road_blocks = [tuple(block) for block in config["road_blocks"]]
    goals = [tuple(goal) for goal in config["goal_pos"]]

    # Print configuration
    print("=== Quantum Walk Configuration ===")
    print("Board State Dimensions:", board_state.shape)
    print("Start Position:", start_pos)
    print("Road Blocks:", road_blocks)
    print("Goals:", goals)

    # Print board visualization
    print_board_state(board_state, start_pos, road_blocks, goals)

    # Run quantum walk
    route = quantum_walk(board_state, start_pos, road_blocks, goals, use_simulator=True)
    print("=== Path Found ===")
    print("Route (2D Array):", route)
    print("===================")


if __name__ == "__main__":
    main()
