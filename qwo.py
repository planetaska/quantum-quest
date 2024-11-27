import sys
import json
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

def create_grid_oracle(n_qubits, valid_moves, forbidden_positions):
    """Create oracle for valid moves on the grid."""
    qc = QuantumCircuit(n_qubits + 1)
    
    # Mark valid moves
    for pos in valid_moves:
        x, y = pos
        bin_pos = format((x << (n_qubits//2)) + y, f'0{n_qubits}b')
        
        # Apply X gates for 0s in binary representation
        for i, bit in enumerate(bin_pos):
            if bit == '0':
                qc.x(i)
        
        # Mark valid state
        qc.mcx(list(range(n_qubits)), n_qubits)
        
        # Uncompute X gates
        for i, bit in enumerate(bin_pos):
            if bit == '0':
                qc.x(i)
    
    # Add phase flip for invalid positions
    for pos in forbidden_positions:
        x, y = pos
        bin_pos = format((x << (n_qubits//2)) + y, f'0{n_qubits}b')
        
        for i, bit in enumerate(bin_pos):
            if bit == '0':
                qc.x(i)
        qc.z(n_qubits)
        for i, bit in enumerate(bin_pos):
            if bit == '0':
                qc.x(i)
                
    return qc

def get_valid_neighbors(pos, board_size, road_blocks):
    """Get valid neighboring positions."""
    x, y = pos
    moves = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    return [(x,y) for x,y in moves if 0 <= x < board_size and 0 <= y < board_size 
            and (x,y) not in road_blocks]

def quantum_search_path(board_size, start, goals, road_blocks):
    path = [start]
    current = start
    remaining_goals = goals.copy()
    
    while remaining_goals:
        # Find closest goal
        closest_goal = min(remaining_goals, 
                         key=lambda g: abs(g[0]-current[0]) + abs(g[1]-current[1]))
        
        # Setup quantum circuit for next move
        n_qubits = 2 * int(np.ceil(np.log2(board_size)))
        qr = QuantumRegister(n_qubits + 1)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        
        # Get valid moves
        valid_moves = get_valid_neighbors(current, board_size, road_blocks)
        
        # Create and apply oracle
        oracle = create_grid_oracle(n_qubits, valid_moves, road_blocks)
        
        # Initialize superposition
        qc.h(range(n_qubits))
        
        # Apply Grover iteration
        iterations = int(np.pi/4 * np.sqrt(len(valid_moves)))
        for _ in range(iterations):
            qc = qc.compose(oracle)
            qc.h(range(n_qubits))
            qc.x(range(n_qubits))
            qc.h(n_qubits-1)
            qc.mcx(list(range(n_qubits-1)), n_qubits-1)
            qc.h(n_qubits-1)
            qc.x(range(n_qubits))
            qc.h(range(n_qubits))
        
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Execute circuit
        backend = AerSimulator()
        result = backend.run(transpile(qc, backend), shots=1000).result()
        counts = result.get_counts()
        
        # Get next position
        next_pos = None
        for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            mid = len(state)//2
            x = int(state[:mid], 2)
            y = int(state[mid:], 2)
            if (x,y) in valid_moves:
                next_pos = (x,y)
                break
        
        if next_pos:
            path.append(next_pos)
            current = next_pos
            if current == closest_goal:
                remaining_goals.remove(closest_goal)
        else:
            break
            
    return path

def main():
    with open(sys.argv[1]) as f:
        config = json.load(f)
    
    board_size = config["board_size"]
    start_pos = tuple(config["start_pos"])
    road_blocks = [tuple(rb) for rb in config["road_blocks"]]
    goals = [tuple(g) for g in config["goal_pos"]][:2]  # Start with 2 goals
    
    path = quantum_search_path(board_size, start_pos, goals, road_blocks)
    
    # Print board
    board = [['.' for _ in range(board_size)] for _ in range(board_size)]
    for x,y in road_blocks:
        board[x][y] = 'R'
    for x,y in goals:
        board[x][y] = 'G'
    for x,y in path:
        if (x,y) not in goals and (x,y) != start_pos:
            board[x][y] = '*'
    board[start_pos[0]][start_pos[1]] = 'S'
    
    print("\n=== Board State ===")
    for row in board:
        print(' '.join(row))
    print("===================")
    
    if path:
        print(f"Path found: {path}")
        print(f"Path length: {len(path)}")

if __name__ == "__main__":
    main()