import sys
import json
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.algorithms import QAOA
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

def create_graph(board_size, road_blocks):
    """Create a graph representation of the grid."""
    G = nx.grid_2d_graph(board_size, board_size)
    # Remove nodes corresponding to roadblocks
    G.remove_nodes_from(road_blocks)
    return G

def get_edge_variables(G):
    """Assign a binary variable to each edge."""
    edge_vars = {}
    idx = 0
    for edge in G.edges():
        edge_vars[edge] = idx
        idx += 1
    return edge_vars

def build_quadratic_program(G, edge_vars, start_pos, goals):
    """Build the quadratic program with constraints."""
    qp = QuadraticProgram()
    num_edges = len(edge_vars)
    num_nodes = len(G.nodes())
    
    # Add binary variables for edges
    for idx in range(num_edges):
        qp.binary_var(name=f'e_{idx}')
    
    # Objective: Minimize the total number of edges used
    linear = {f'e_{idx}': 1 for idx in range(num_edges)}
    qp.minimize(linear=linear)
    
    # Flow constraints
    for node in G.nodes():
        in_edges = []
        out_edges = []
        for edge in G.edges(node):
            idx = edge_vars[edge if edge[0]==node else (edge[1], edge[0])]
            if edge[0] == node:
                out_edges.append(f'e_{idx}')
            else:
                in_edges.append(f'e_{idx}')
        
        # Start position: net flow out is 1
        if node == start_pos:
            coeffs = {var: 1 for var in out_edges}
            coeffs.update({var: -1 for var in in_edges})
            qp.linear_constraint(linear=coeffs, sense='==', rhs=1, name=f'flow_start_{node}')
        # Goal nodes: net flow in is 1
        elif node in goals:
            coeffs = {var: 1 for var in in_edges}
            coeffs.update({var: -1 for var in out_edges})
            qp.linear_constraint(linear=coeffs, sense='==', rhs=1, name=f'flow_goal_{node}')
        # Intermediate nodes: net flow zero
        else:
            coeffs = {var: 1 for var in in_edges}
            coeffs.update({var: -1 for var in out_edges})
            qp.linear_constraint(linear=coeffs, sense=='==', rhs=0, name=f'flow_{node}')
    
    return qp

def solve_qaoa(qp):
    """Solve the quadratic program using QAOA."""
    algorithm_globals.random_seed = 42
    backend = AerSimulator()
    quantum_instance = QuantumInstance(backend=backend)
    cobyla = COBYLA(maxiter=100)
    qaoa = QAOA(optimizer=cobyla, reps=1, quantum_instance=quantum_instance)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    return result

def extract_path_from_result(result, G, edge_vars):
    """Extract the path from the optimization result."""
    selected_edges = []
    for edge, idx in edge_vars.items():
        if result.variables_dict[f'e_{idx}'] > 0.5:
            selected_edges.append(edge)
    # Build the path from selected edges
    path = []
    if selected_edges:
        # Build a multigraph to account for possible cycles
        MG = nx.MultiGraph()
        MG.add_edges_from(selected_edges)
        # Find connected components
        for component in nx.connected_components(MG):
            subgraph = MG.subgraph(component)
            # Try to find a path that covers all nodes in the subgraph
            try:
                path_nodes = list(nx.shortest_path(subgraph, source=start_pos))
                path = path_nodes
                break  # Stop after finding one path
            except nx.NetworkXError:
                # No path exists
                pass
    return path

def print_board(board_size, start_pos, road_blocks, goals, path):
    board = [['.' for _ in range(board_size)] for _ in range(board_size)]
    
    for x, y in road_blocks:
        board[x][y] = 'R'
    for x, y in goals:
        board[x][y] = 'G'
    for x, y in path:
        if (x, y) not in goals and (x, y) != start_pos:
            board[x][y] = '*'
    board[start_pos[0]][start_pos[1]] = 'S'
    
    print("\n=== Board State ===")
    for row in board:
        print(' '.join(row))
    print("===================")

def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.json"
    
    with open(config_file, "r") as file:
        config = json.load(file)
    
    board_size = config["board_size"]
    start_pos = tuple(config["start_pos"])
    road_blocks = [tuple(rb) for rb in config["road_blocks"]]
    goals = [tuple(g) for g in config["goal_pos"]]
    
    G = create_graph(board_size, road_blocks)
    edge_vars = get_edge_variables(G)
    qp = build_quadratic_program(G, edge_vars, start_pos, goals)
    result = solve_qaoa(qp)
    path = extract_path_from_result(result, G, edge_vars)
    
    print_board(board_size, start_pos, road_blocks, goals, path)
    
    if path:
        print(f"Path found: {path}")
        print(f"Path length: {len(path)}")
    else:
        print("No valid path found")

if __name__ == "__main__":
    main()
