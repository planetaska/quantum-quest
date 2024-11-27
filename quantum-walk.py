from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import numpy as np
import matplotlib.pyplot as plt


def index_to_binary(index, num_qubits):
    """Convert grid index to binary representation."""
    return format(index, f'0{num_qubits}b')


def binary_to_index(binary_str):
    """Convert binary string to an integer index."""
    return int(binary_str, 2)


def index_to_coordinates(index, grid_size):
    """Convert a flattened index to grid coordinates."""
    return divmod(index, grid_size)


def add_oracle(qc, goals, num_qubits):
    """Add an oracle that marks the goal states."""
    for goal in goals:
        binary_goal = index_to_binary(goal, num_qubits - 1)
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
    # Simplified cyclic shift: For an actual grid, this would encode movement rules.
    for i in range(num_qubits - 1):
        qc.cx(i, num_qubits - 1)  # Move based on the state of the coin qubit


def quantum_walk(grid_size, goals, iterations=3):
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
            route.append(coord)

    # Sort the route based on the grid traversal pattern (optional for clarity)
    route = sorted(set(route))  # Deduplicate and sort

    # Plot measurement histogram
    plt.figure(figsize=(8, 6))
    plot_histogram(counts)
    plt.show()

    print("Route (2D Array):", route)
    return route


# Example usage
grid_size = 8  # 8x8 grid
goals = [7, 15]  # Example goal indices in a flattened array
route = quantum_walk(grid_size, goals)
