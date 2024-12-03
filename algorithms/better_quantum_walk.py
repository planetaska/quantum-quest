from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import json
import sys
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def create_oracle(qc, goals, grid_size, num_position_qubits):
    """Create oracle that marks goal states with a phase flip."""
    for goal in goals:
        # Convert goal position to binary representation
        goal_index = goal[0] * grid_size + goal[1]
        goal_bin = format(goal_index, f'0{num_position_qubits}b')
        
        # Mark goal state with Z gates on 0 bits and controls on 1 bits
        for i, bit in enumerate(goal_bin):
            if bit == '0':
                qc.x(i)  # Flip for control on 0
        
        # Apply controlled-Z to mark goal state
        qc.h(num_position_qubits)  # Last qubit is flag
        qc.mcx(list(range(num_position_qubits)), num_position_qubits)  # Multi-controlled X
        qc.h(num_position_qubits)
        
        # Uncompute X gates
        for i, bit in enumerate(goal_bin):
            if bit == '0':
                qc.x(i)

def diffusion_operator(qc, num_position_qubits):
    """Apply diffusion operator for amplitude amplification."""
    # Apply H gates to all position qubits
    for i in range(num_position_qubits):
        qc.h(i)
    
    # Apply X gates to all position qubits
    for i in range(num_position_qubits):
        qc.x(i)
        
    # Apply multi-controlled Z
    qc.h(num_position_qubits-1)
    qc.mcx(list(range(num_position_qubits-1)), num_position_qubits-1)
    qc.h(num_position_qubits-1)
    
    # Uncompute X gates
    for i in range(num_position_qubits):
        qc.x(i)
        
    # Uncompute H gates
    for i in range(num_position_qubits):
        qc.h(i)

def get_valid_neighbors(pos, grid_size, road_blocks):
    """Get valid adjacent positions that aren't roadblocks."""
    row, col = pos
    neighbors = []
    for dr, dc in [(-1,0), (0,1), (1,0), (0,-1)]:  # Up, Right, Down, Left
        new_row, new_col = row + dr, col + dc
        new_pos = (new_row, new_col)
        if (0 <= new_row < grid_size and 
            0 <= new_col < grid_size and 
            new_pos not in road_blocks):
            neighbors.append(new_pos)
    return neighbors

def draw_circuit(qc, filename="quantum_circuit.png"):
    """Draw quantum circuit on a single line."""
    try:
        fig, ax = plt.subplots(figsize=(40, 6))
        ax.axis('off')
        
        style = {
            'compress': False,
            'fold': 0,
            'wire_order': list(range(qc.num_qubits))
        }
        
        circuit_diagram = circuit_drawer(
            qc,
            output='mpl',
            style=style,
            interactive=False,
            justify='none',
            idle_wires=False,
            scale=0.8
        )
        
        circuit_diagram.figure.set_size_inches(40, 6)
        circuit_diagram.figure.savefig(
            filename,
            bbox_inches='tight',
            dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error saving circuit diagram: {e}")

def quantum_walk(grid_size, goals, road_blocks, start_pos=(0,0)):
    """Quantum walk to find path visiting all goals."""
    # Initialize complete path
    complete_path = [start_pos]
    remaining_goals = goals.copy()
    current_pos = start_pos
    
    # Calculate number of iterations based on grid size
    N = grid_size * grid_size
    num_iterations = int(np.pi/4 * np.sqrt(N))
    print(f"Using {num_iterations} amplitude amplification iterations")
    
    # Find path through all goals
    while remaining_goals:
        # Create quantum circuit for finding next goal
        num_position_qubits = int(np.ceil(np.log2(N)))
        qc = QuantumCircuit(num_position_qubits + 1, num_position_qubits)
        
        # Initialize superposition
        qc.h(range(num_position_qubits))
        
        # Apply amplitude amplification iterations
        for _ in range(num_iterations):
            # Oracle marks closest remaining goal
            closest_goal = min(remaining_goals, 
                             key=lambda g: abs(g[0] - current_pos[0]) + abs(g[1] - current_pos[1]))
            create_oracle(qc, [closest_goal], grid_size, num_position_qubits)
            
            # Diffusion operator
            diffusion_operator(qc, num_position_qubits)
            
            # Quantum walk operations
            for i in range(num_position_qubits):
                qc.h(i)
                qc.z(i)
                qc.h(i)
            for i in range(num_position_qubits):
                qc.cx(i, num_position_qubits)
        
        # Measure and run circuit
        qc.measure(range(num_position_qubits), range(num_position_qubits))
        
        simulator = AerSimulator()
        job = simulator.run(transpile(qc, simulator), shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Get measured position
        most_frequent = max(counts.items(), key=lambda x: x[1])[0]
        position = int(most_frequent, 2)
        row, col = position // grid_size, position % grid_size
        
        # Find path to measured position (avoiding obstacles)
        while current_pos != (row, col):
            neighbors = get_valid_neighbors(current_pos, grid_size, road_blocks)
            if not neighbors:
                break
                
            # Move towards measured position
            next_pos = min(neighbors, key=lambda x: 
                         abs(x[0] - row) + abs(x[1] - col))
            current_pos = next_pos
            complete_path.append(current_pos)
            
            # Check if we hit a goal
            if current_pos in remaining_goals:
                remaining_goals.remove(current_pos)
                break
            
            # Prevent infinite loops
            if len(complete_path) > grid_size * grid_size * len(goals):
                return complete_path
                
    return complete_path

def print_board_state(board_size, start_pos, road_blocks, goals, path=None):
    """Print visual representation of the board state."""
    board = [["." for _ in range(board_size)] for _ in range(board_size)]
    
    # Mark roadblocks
    for r, c in road_blocks:
        board[r][c] = "R"
    
    # Mark path
    if path:
        for r, c in path:
            if (r, c) != start_pos and (r, c) not in goals:
                board[r][c] = "*"
    
    # Mark goals
    for r, c in goals:
        board[r][c] = "G"
    
    # Mark start
    sr, sc = start_pos
    board[sr][sc] = "S"
    
    print("\n=== Board State ===")
    for row in board:
        print(" ".join(row))
    print("===================")

def main():
    if len(sys.argv) < 2:
        print("Usage: python quantum_walk.py <config_file>")
        sys.exit(1)
    
    # Load configuration
    with open(sys.argv[1], "r") as file:
        config = json.load(file)
    
    board_size = config.get("board_size", 8)
    start_pos = tuple(config.get("start_pos", [0, 0]))
    road_blocks = [tuple(block) for block in config.get("road_blocks", [])]
    goals = [tuple(goal) for goal in config.get("goal_pos", [[board_size-1, board_size-1]])]
    
    print("=== Quantum Walk Algorithm Configuration ===")
    print("Board Size:", board_size)
    print("Start Position:", start_pos)
    print("Road Blocks:", road_blocks)
    print("Goals:", goals)
    print("==========================================")
    
    # Run quantum walk
    route = quantum_walk(board_size, goals, road_blocks, start_pos)
    
    # Display results
    print_board_state(board_size, start_pos, road_blocks, goals, route)
    print("Route:", route)
    print("Path length:", len(route))

if __name__ == "__main__":
    main()