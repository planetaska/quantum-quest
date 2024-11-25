from qiskit import QuantumCircuit, transpile
from qiskit_aer.aerprovider import AerSimulator
import numpy as np
import json

# Receiving command line parameters?
# Or using a text file containing the parameters maybe in JSON format?
import sys
args = sys.argv



def quantum_walk(board_state=None, start_pos=(0, 0), road_blocks=None, goal_pos=None, use_simulator=True):
    if board_state is None:
        board_state = np.zeros((8, 8))

    if road_blocks is None:
        road_blocks = []

    def create_circuit(dim):
        # Create a quantum walk circuit for given dimensions.
        n_qubits = int(np.ceil(np.log2(dim ** 2)))
        qc = QuantumCircuit(n_qubits, n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        qc.measure_all()
        return qc

    def get_next_position(current_pos, n):
        # Determine the next position based on the measurement outcome.
        next_position = ((current_pos[0] + n // len(board_state)) % len(board_state),
                         (current_pos[1] + n % len(board_state[0])) % len(board_state[0]))
        return next_position

    dim = len(board_state)
    qc = create_circuit(dim)
    qc.draw("mpl", filename="circuit_qw.png")

    backend = AerSimulator()

    path = [list(start_pos)]
    current_pos = start_pos

    # Uncomment the for... loop to test just a few steps
    while current_pos != goal_pos:
      # for i in range(5):
        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc).result()
        counts = result.get_counts(qc)
        # print(counts)

        # Get the move with the highest probability from the counts.
        next_move = max(counts, key=counts.get)
        # print(next_move)
        
        # Convert the binary string (next_move) to a decimal integer.
        decimal_move = int(next_move.split(' ', 1)[0], 2)
        # print(decimal_move)

        next_pos = get_next_position(current_pos, decimal_move)

        if next_pos in road_blocks:
            continue

        path.append(list(next_pos))  # Append as a list for consistency in 2D array.
        current_pos = next_pos

    return path


# Example usage
# Load configuration
def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.json"


    with open("config.json", "r") as file:
        config = json.load(file)

    # Create board state dynamically based on size
    board_size = config["board_size"]
    board_state = np.zeros((board_size, board_size))

    # Extract other parameters
    start_pos = tuple(config["start_pos"])
    road_blocks = [tuple(block) for block in config["road_blocks"]]
    goal_pos = tuple(config["goal_pos"])

    print("Board State:\n", board_state)
    print("Start Position:", start_pos)
    print("Road Blocks:", road_blocks)
    print("Goal Position:", goal_pos)



    # TODO: format the output so Godot can read it
    # Candidate format: something like a 2D array
    # [[0, 0], [0, 2], ... [7, 7]]
    route = quantum_walk(board_state, start_pos, road_blocks, goal_pos, use_simulator=True)
    print("Route:", route)
    return route

if __name__ == "__main__":
    main()