# Import necessary Qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer.aerprovider import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np

from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt


def grover_pathfinding(board, start, goal, use_real_device=False):
    if use_real_device:
        # If you did not previously save your credentials, use the following line instead:
        # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False, operational=True)
    else:
        backend = AerSimulator()
        
    # Function to determine if two cells are adjacent in a square grid
    def are_adjacent(i, j, size):
        xi, yi = divmod(i, size)
        xj, yj = divmod(j, size)
        return (xi == xj and abs(yi - yj) == 1) or (yi == yj and abs(xi - xj) == 1)

    # Define the oracle
    def oracle(circuit, register):
        num_qubits = len(register)
        board_size = int(np.sqrt(num_qubits))  # assuming num_qubits is a square

        goal_index = goal[0] * board_size + goal[1]
        encoded_gl_pos = list(map(int, bin(goal_index)[2:].zfill(num_qubits)))
        print("Encoded goal position:", encoded_gl_pos)
        # oracle_circuit.z(bin(goal_index)[2:].zfill(n_qubits))
        for idx in range(len(encoded_gl_pos)):
            if encoded_gl_pos[idx] == 1:
                circuit.z(idx)

    # Grover diffusion operator
    def diffuser(circuit, register):
        circuit.h(register)
        circuit.x(register)
        circuit.h(register[-1])
        circuit.mcx(register[:-1], register[-1])  # multi-controlled X gate
        circuit.h(register[-1])
        circuit.x(register)
        circuit.h(register)

    # Define the quantum circuit
    num_qubits = len(board)
    register = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(register)

    # Initial Hadamard gate
    circuit.h(register)

    # Amplification iterations (oracle + diffuser) for √N times 
    iterations = int(np.sqrt(len(board)))
    print(f"Amplification iterations: {iterations}")
    for _ in range(iterations):
        oracle(circuit, register)
        diffuser(circuit, register)

    # Measure the result
    circuit.measure_all()
    circuit.barrier()

    circuit.draw(output='mpl')

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(circuit)

    sampler = Sampler(backend)
    job = sampler.run([isa_qc])
    result = job.result()
    # print(result[0].data.meas.get_counts())
    # return None
    counts = result[0].data.meas.get_counts()
    # counts = result[0].data.c.get_counts()

    # Perform the quantum computation
    # transpiled_circuit = transpile(circuit, backend)
    # qobj = assemble(transpiled_circuit)
    # result = backend.run(qobj).result()
    # counts = result.get_counts(circuit)

    # Calculate the path coordinates from the bitstring
    
    def bitstring_to_coordinates(bitstring, board_size):
        path = []
        for idx, bit in enumerate(bitstring):
            if bit == '1':  # assuming '1' indicates part of the path
                x = idx // board_size
                y = idx % board_size
                path.append((x, y))
        return path

    # Extract the solution (path) from the result
    most_common_bitstring = max(counts, key=counts.get)
    print(
        f"Most common bitstring: {most_common_bitstring}, Count: {counts[most_common_bitstring]}"
    )
    plot_histogram(counts)
    plt.show()

    path = bitstring_to_coordinates(most_common_bitstring, len(board))
    # Extract the solution (path) from the result
    # most_common_bitstring = max(counts, key=counts.get)
    # path = [int(bit) for bit in most_common_bitstring]

    return path


# Example usage
board = [[0] * 8 for _ in range(8)]  # 8x8 grid
start = (0, 0)
goal = (7, 7)

# Finding path using simulator (set use_real_device to True for using real quantum device)
path = grover_pathfinding(board, start, goal, use_real_device=False)
print("Path from start to goal:", path)
