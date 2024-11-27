import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.aerprovider import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.result import Counts
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService

def run_grovers_algorithm(board_size=(8, 8), start=(0, 0), road_blocks=None, goal=(7, 7), use_simulator=True):
    if road_blocks is None:
        road_blocks = []

    if use_simulator:
        backend = AerSimulator()
    else:
        # If you did not previously save your credentials, use the following line instead:
        # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False, operational=True)

    # Define the oracle
    def oracle_for_pathfinding(board_size, start, road_blocks, goal):
        # For an 8x8 board, the total number of positions is 64, which means we need 6 qubits (2^6)
        n_qubits = int(np.log2(board_size[0] * board_size[1]))
        oracle_circuit = QuantumCircuit(n_qubits)

        for block in road_blocks:
            roadblock_index = block[0] * board_size[1] + block[1]
            # print(roadblock_index)
            # Convert the roadblock index to a binary string and pad with leading zeros to fill the number of qubits
            encoded_rb_pos = list(map(int, bin(roadblock_index)[2:].zfill(n_qubits)))
            print("Encoded road block position:", encoded_rb_pos)
            # oracle_circuit.x(bin(roadblock_index)[2:].zfill(n_qubits))
            for idx in range(len(encoded_rb_pos)):
                if encoded_rb_pos[idx] == 1:
                    oracle_circuit.x(idx)

        # Convert the goal coordinate from a 2D board position to a linear index.
        goal_index = goal[0] * board_size[1] + goal[1]
        encoded_gl_pos = list(map(int, bin(goal_index)[2:].zfill(n_qubits)))
        print("Encoded goal position:", encoded_gl_pos)
        # oracle_circuit.z(bin(goal_index)[2:].zfill(n_qubits))
        for idx in range(len(encoded_gl_pos)):
            if encoded_gl_pos[idx] == 1:
                oracle_circuit.z(idx)

        oracle_circuit.draw(output='mpl', filename='circuit_oracle.png')

        return oracle_circuit

    n_qubits = int(np.log2(board_size[0] * board_size[1]))
    oracle = oracle_for_pathfinding(board_size, start, road_blocks, goal)
    # return None

    grover_op = GroverOperator(oracle=oracle, insert_barriers=True)
    grover_op.decompose().draw(output='mpl', filename='circuit_grover.png')
    # return None

    goal_index = goal[0] * board_size[1] + goal[1]
    encoded_gl_pos = list(map(int, bin(goal_index)[2:].zfill(n_qubits)))
    def encoded_goal(position):
        print("Iterating goal position:", encoded_gl_pos)
        # oracle_circuit.x(bin(roadblock_index)[2:].zfill(n_qubits))
        return encoded_gl_pos

    # goal_index = goal[0] * board_size[1] + goal[1]
    # goal_circuit = QuantumCircuit(n_qubits)
    # encoded_gl_pos = list(map(int, bin(goal_index)[2:].zfill(n_qubits)))
    # print("Encoded goal position:", encoded_gl_pos)
    # # oracle_circuit.z(bin(goal_index)[2:].zfill(n_qubits))
    # for idx in range(len(encoded_gl_pos)):
    #     if encoded_gl_pos[idx] == 1:
    #         goal_circuit.z(idx)

    # Using the built-in Grover's algorithm implementation
    problem = AmplificationProblem(oracle, is_good_state=encoded_goal)
    grover = Grover(sampler=Sampler())
    # grover = Grover(quantum_instance=backend)

    # return None

    result = grover.amplify(problem)
    # job = grover.amplify(problem)
    # print(f">>> Job ID: {job.job_id()}")
    # return None
    # result = job.result()

    print(result)
    return None

    # Converting measured state back to coordinates
    measured_state = Counts(result.circuit_results[0]).most_frequent
    # counts = result.get_counts(result)

    path_index = int(measured_state, 2)
    path_coordinates = (path_index // board_size[1], path_index % board_size[1])

    return path_coordinates


board_size = (8, 8)
start = (0, 0)
road_blocks = [(1, 1), (1, 2), (2, 2)]
goal = (7, 7)
result = run_grovers_algorithm(board_size, start, road_blocks, goal, use_simulator=True)
print("The path coordinates to the goal are:", result)
