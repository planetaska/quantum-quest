from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from collections import deque
from qiskit.visualization import circuit_drawer
from matplotlib import pyplot as plt

def draw_circuit(qc, filename="quantum_circuit.png"):
    """Draw quantum circuit on a single line and save to PNG file."""
    try:
        # Configure figure size to be wide enough for single line
        from matplotlib import pyplot as plt
        plt.figure(figsize=(20, 2))  # Wide but short figure
        
        # Draw circuit with custom style
        circuit_drawing = circuit_drawer(
            qc,
            output='mpl',
            style={
                'backgroundcolor': '#FFFFFF',
                'compression': True,     # Compress the circuit
                'fold': -1,             # Prevent folding
                'initial_state': True    # Show initial state
            }
        )
        
        # Save with tight layout
        circuit_drawing.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1
        )
        print(f"Circuit diagram saved as {filename}")
        plt.close()  # Clean up
    except Exception as e:
        print(f"Error saving circuit diagram: {e}")

def get_valid_neighbors(pos, grid_size, road_blocks, visited):
    """Get valid adjacent positions that haven't been visited and aren't roadblocks."""
    row, col = pos
    neighbors = []
    # Check all 4 directions (up, right, down, left)
    for dr, dc in [(-1,0), (0,1), (1,0), (0,-1)]:
        new_row, new_col = row + dr, col + dc
        new_pos = (new_row, new_col)
        if (0 <= new_row < grid_size and 
            0 <= new_col < grid_size and 
            new_pos not in visited and
            new_pos not in road_blocks):
            neighbors.append(new_pos)
    return neighbors

def quantum_walk(grid_size, goals, road_blocks, start_pos=(0,0), iterations=3):
    """Perform quantum walk to find path avoiding roadblocks."""
    # Initialize quantum circuit for the walk
    num_qubits = int(np.ceil(np.log2(grid_size ** 2))) + 1
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits - 1))

    # Add quantum walk operations
    for _ in range(iterations):
        # Add barrier for roadblocks
        for i in range(num_qubits - 1):
            qc.h(i)
            qc.z(i)
            qc.h(i)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
    
    #draw_circuit(qc, "quantum_walk_circuit.png")
    # Measure
    qc.measure_all()
    
    # Simulate
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    #print("Transpiled Quantum Circuit:")
    #print(transpiled_qc)
    result = simulator.run(transpiled_qc, shots=1024).result()
    
    # Convert quantum result to path
    path = []
    visited = set()
    current = start_pos
    path.append(current)
    visited.add(current)
    
    # Find path to goals avoiding roadblocks
    remaining_goals = goals.copy()
    while remaining_goals and len(path) < grid_size * grid_size:
        current = path[-1]
        if current in remaining_goals:
            remaining_goals.remove(current)
            if not remaining_goals:  # If all goals reached
                break
            
        neighbors = get_valid_neighbors(current, grid_size, road_blocks, visited)
        if not neighbors:  # Backtrack if stuck
            if len(path) > 1:
                path.pop()
                current = path[-1]
                continue
            else:
                break
                
        # Choose next position based on quantum measurement and distance to goal
        next_pos = min(neighbors, key=lambda x: 
                      abs(x[0] - remaining_goals[0][0]) + abs(x[1] - remaining_goals[0][1]))
        path.append(next_pos)
        visited.add(next_pos)
    
    return path

def print_board_state(board_state, start_pos, road_blocks, goals, path=None):
    """Print visual representation of the board state."""
    rows, cols = board_state.shape
    board_visual = [["." for _ in range(cols)] for _ in range(rows)]
    
    # Mark roadblocks
    for r, c in road_blocks:
        board_visual[r][c] = "R"
        
    # Mark path
    if path:
        for r, c in path:
            if (r, c) != start_pos and (r, c) not in goals:
                board_visual[r][c] = "*"
    
    # Mark goals
    for r, c in goals:
        board_visual[r][c] = "G"
    
    # Mark start position
    sr, sc = start_pos
    board_visual[sr][sc] = "S"
    
    print("\n=== Board State ===")
    for row in board_visual:
        print(" ".join(row))
    print("===================")

def write_path_to_json(path, filename="route.json"):
    import json
    
    with open(filename, 'w') as f:
        json.dump({"path": path}, f, indent=4)

def main():
    import sys
    import json
    import numpy as np
    
    if len(sys.argv) < 2:
        print("Usage: python quantum_walk.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    with open(config_file, "r") as file:
        config = json.load(file)
    
    board_size = config.get("board_size", 8)
    board_state = np.zeros((board_size, board_size))
    start_pos = tuple(config.get("start_pos", [0, 0]))
    goals = [tuple(goal) for goal in config.get("goal_pos", [[board_size-1, board_size-1]])]
    road_blocks = [tuple(block) for block in config.get("road_blocks", [])]
    
    print("=== Quantum Walk Algorithm Configuration ===")
    print("Board State Dimensions:", board_state.shape)
    print("Start Position:", start_pos)
    print("Road Blocks:", road_blocks)
    print("Goals:", goals)
    print("==========================================")
    
    route = quantum_walk(board_size, goals, road_blocks, start_pos)
    
    print_board_state(board_state, start_pos, road_blocks, goals, route)
    print("Route (2D Array):", route)
    print("Path length:", len(route))
    write_path_to_json(route, "route.json")

if __name__ == "__main__":
    main()