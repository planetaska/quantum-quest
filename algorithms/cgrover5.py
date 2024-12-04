import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer.aerprovider import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit_algorithms import AmplificationProblem, Grover

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

import sys
import json

class QuantumPathFinder:
    def __init__(self, board_size=(8, 8), start=(0, 0), goal=None, roadblocks=None, backend_type='simulator'):
        """
        Initialize Quantum Path Finding algorithm

        Args:
            board_size (tuple): Dimensions of the grid (rows, columns)
            start (tuple): Starting position coordinates
            goal (tuple): Goal position coordinates
            roadblocks (list): List of coordinate tuples representing blocked positions
            backend_type (str): 'simulator' or 'real'
        """
        self.board_size = board_size
        self.start = start
        self.goal = goal or (board_size[0] - 1, board_size[1] - 1)
        self.roadblocks = roadblocks or []

        # Calculate overall direction to goal
        self.goal_direction = (
            np.sign(self.goal[0] - self.start[0]),  # row direction
            np.sign(self.goal[1] - self.start[1])  # column direction
        )

        # Validate inputs
        self._validate_board_state()

        # Backend selection
        if backend_type == 'simulator':
            self.backend = AerSimulator()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            # If you did not previously save your credentials, use the following line instead:
            # service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")

            self.backend = service.least_busy(operational=True, simulator=False)

    def _validate_board_state(self):
        """Validate input parameters for board state"""
        assert self.start[0] < self.board_size[0] and self.start[1] < self.board_size[1], \
            "Start position is outside board dimensions"
        assert self.goal[0] < self.board_size[0] and self.goal[1] < self.board_size[1], \
            "Goal position is outside board dimensions"

        for block in self.roadblocks:
            assert block[0] < self.board_size[0] and block[1] < self.board_size[1], \
                f"Roadblock {block} is outside board dimensions"

        assert self.start not in self.roadblocks, "Start position cannot be a roadblock"
        assert self.goal not in self.roadblocks, "Goal position cannot be a roadblock"

    def _is_valid_move(self, current, next_pos, previous_pos=None):
        """
        Check if move between two positions is valid

        Args:
            current (tuple): Current position
            next_pos (tuple): Next position to check
            previous_pos (tuple, optional): Previous position to prevent backtracking

        Returns:
            bool: Whether the move is valid
        """
        # Check if next position is within board
        if (next_pos[0] < 0 or next_pos[0] >= self.board_size[0] or
                next_pos[1] < 0 or next_pos[1] >= self.board_size[1]):
            return False

        # Check if next position is not a roadblock
        if next_pos in self.roadblocks:
            return False

        # Check if move is to an adjacent cell (up, down, left, right)
        is_adjacent = (abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1])) == 1

        # Prevent backtracking
        if previous_pos and next_pos == previous_pos:
            return False

        return is_adjacent

    def _get_possible_moves(self, current, previous=None):
        """
        Generate possible moves from current position

        Args:
            current (tuple): Current position
            previous (tuple, optional): Previous position to prevent backtracking

        Returns:
            list: List of valid moves
        """
        moves = [
            (current[0] + 1, current[1]),  # down
            (current[0] - 1, current[1]),  # up
            (current[0], current[1] + 1),  # right
            (current[0], current[1] - 1)  # left
        ]
        return [move for move in moves if self._is_valid_move(current, move, previous)]

    def _calculate_move_score(self, current, move):
        """
        Calculate a score for a potential move

        Args:
            current (tuple): Current position
            move (tuple): Potential next move

        Returns:
            float: Score for the move
        """
        # Distance to goal (Manhattan distance)
        goal_distance = abs(move[0] - self.goal[0]) + abs(move[1] - self.goal[1])

        # Directional alignment bonus
        move_direction = (
            np.sign(move[0] - current[0]),  # row direction
            np.sign(move[1] - current[1])  # column direction
        )

        # Bonus for moves aligned with overall goal direction
        direction_bonus = (
            3.5 if move_direction == self.goal_direction else
            2.2 if move_direction[0] == self.goal_direction[0] or
                   move_direction[1] == self.goal_direction[1] else
            1.0
        )

        # Proximity bonus
        proximity_bonus = 1 / (goal_distance + 1)

        return proximity_bonus * direction_bonus

    def _create_grover_oracle(self, current, previous, possible_moves):
        """
        Create a quantum oracle for Grover's algorithm to select moves

        Args:
            current (tuple): Current position
            previous (tuple): Previous position
            possible_moves (list): List of possible next moves

        Returns:
            QuantumCircuit: Oracle circuit
        """
        # Determine number of qubits needed to represent moves
        num_moves = len(possible_moves)
        num_qubits = int(np.ceil(np.log2(num_moves)))

        # Create quantum register
        qr = QuantumRegister(num_qubits)
        oracle = QuantumCircuit(qr)

        # Calculate scores for moves
        move_scores = [self._calculate_move_score(current, move) for move in possible_moves]
        max_score = max(move_scores)

        # Mark states corresponding to moves with highest scores
        for i, score in enumerate(move_scores):
            # if score >= max_score * 0.9:  # Allow some tolerance
            if score >= max_score:
                # Convert move index to binary and apply oracle marking
                binary_index = format(i, f'0{num_qubits}b')
                for j, bit in enumerate(binary_index):
                    if bit == '0':
                        oracle.x(qr[j])

        return oracle

    def select_move_with_grovers(self, current, previous, possible_moves):
        """
        Use Grover's algorithm to select the next move

        Args:
            current (tuple): Current position
            previous (tuple): Previous position
            possible_moves (list): List of possible next moves

        Returns:
            tuple: Selected move
        """
        # If only one possible move, return it
        if len(possible_moves) == 1:
            return possible_moves[0]

        # Determine number of qubits needed to represent moves
        num_moves = len(possible_moves)
        num_qubits = int(np.ceil(np.log2(num_moves)))

        # Create quantum circuit
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        qc = QuantumCircuit(qr, cr)

        # Apply Hadamard gates to create uniform superposition
        qc.h(qr)

        # Create Grover oracle
        oracle = self._create_grover_oracle(current, previous, possible_moves)

        # Create Grover operator
        grover_op = GroverOperator(oracle)

        # Optimal number of iterations
        num_iterations = int(np.pi / 4 * np.sqrt(2 ** num_qubits))

        # Apply Grover iterations
        for _ in range(num_iterations):
            qc.compose(grover_op, inplace=True)

        # Measure
        qc.measure(qr, cr)

        # circuit optimization
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        isa_qc = pm.run(qc)

        # run with sampler
        sampler = Sampler(self.backend)
        job = sampler.run([isa_qc])
        result = job.result()

        # show the result
        # print(f" > Counts: {result[0].join_data().get_counts()}")
        counts = result[0].join_data().get_counts()
        # print(f" > Counts: {counts}")

        # Execute
        # job = qiskit.execute(qc, self.backend, shots=1024)
        # result = job.result()
        # counts = result.get_counts(qc)

        # Find most probable state
        most_probable_state = max(counts, key=counts.get)
        move_index = int(most_probable_state, 2)
        print(f" > Most probable state: {most_probable_state}, index: {move_index}")

        # Ensure the index is within possible moves
        move_index = move_index % len(possible_moves)

        return possible_moves[move_index]

    def find_path(self, max_path_length=None):
        """
        Find a path from start to goal using Grover's algorithm

        Args:
            max_path_length (int): Maximum allowed path length

        Returns:
            List of coordinates representing the path
        """
        if max_path_length is None:
            max_path_length = (self.board_size[0] + self.board_size[1]) * 2

        # Initial path starts with start position
        current_path = [self.start]
        current_pos = self.start
        previous_pos = None

        while current_pos != self.goal and len(current_path) < max_path_length:
            # Get possible next moves
            possible_moves = self._get_possible_moves(current_pos, previous_pos)

            if not possible_moves:
                # No valid moves, backtrack
                current_path.pop()
                if not current_path:
                    return None  # No path found
                previous_pos = current_path[-2] if len(current_path) > 1 else None
                current_pos = current_path[-1]
                continue

            # Use Grover's algorithm to select next move
            next_move = self.select_move_with_grovers(current_pos, previous_pos, possible_moves)

            # Update path
            previous_pos = current_pos
            current_path.append(next_move)
            current_pos = next_move

        return current_path if current_pos == self.goal else None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cgrover5.py <config_file>")
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
    # board_state = np.zeros((board_size, board_size))
    start_pos = tuple(config.get("start_pos", [0, 0]))
    road_blocks = [tuple(block) for block in config.get("road_blocks", [])]
    goals = [tuple(goal) for goal in config.get("goal_pos", [[board_size - 1, board_size - 1]])]

    path_finder = QuantumPathFinder(
        board_size=(board_size, board_size),
        start=start_pos,
        goal=goals[0],
        roadblocks=road_blocks,
        backend_type='simulator'
    )

    # Run multiple iterations to improve chance of getting a reasonable result
    max_iteration = 10
    current_iteration = 0
    path = None
    while path is None and current_iteration < max_iteration:
        current_iteration += 1
        path = path_finder.find_path()

    # path = path_finder.find_path()

    # Print the route
    if path:
        print(f"Found path: {path}")

        # Save route to JSON file
        with open("route.json", "w") as json_file:
            json.dump({"path": path}, json_file, indent=4)
        print("Path saved to route.json")

    else:
        print("No path found within iteration limits.")


# Example usage
if __name__ == "__main__":
    main()
    # Example configuration
    # roadblocks = [(1, 2), (2, 3), (3, 4)]
    # path_finder = QuantumPathFinder(
    #     board_size=(8, 8),
    #     start=(0, 0),
    #     goal=(7, 7),
    #     roadblocks=roadblocks,
    #     backend_type='simulator'
    # )
    #
    # path = path_finder.find_path()
    # print(f"Found path: {path}")