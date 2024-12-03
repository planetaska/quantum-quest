import numpy as np
import matplotlib.pyplot as plt

try:
    import cplex
    from cplex.exceptions import CplexError
except:
    print("Warning: Cplex not found.")
import math

from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit import transpile
import sys
import json

# Initialize the problem by defining the parameters
n = 3  # number of nodes + depot (n+1)
K = 1  # number of vehicles

class Initializer:
    def __init__(self, n):
        self.n = n

    def generate_instance(self, board_file=None):
        import json
        
        if board_file:
            with open(board_file, 'r') as f:
                board = json.load(f)
            
            coords = [board["start_pos"]] + board["goal_pos"]
            xc = [coord[0] for coord in coords]
            yc = [coord[1] for coord in coords]
        else:
            np.random.seed(1543)
            xc = [0, 5, 2]
            yc = [0, 5, 7]

        # Use length of coordinates for instance size
        m = len(xc)  
        instance = np.zeros([m, m])
        for ii in range(0, m):
            for jj in range(ii + 1, m):
                instance[ii, jj] = (xc[ii] - xc[jj]) ** 2 + (yc[ii] - yc[jj]) ** 2
                instance[jj, ii] = instance[ii, jj]

        return xc, yc, instance


# Initialize the problem by randomly generating the instance
config_file = sys.argv[1]
initializer = Initializer(n)
xc, yc, instance = initializer.generate_instance(config_file)
n = len(xc)

#Print Xc Yc and Instance
print("Xc: ", xc)
print("Yc: ", yc)
print("Instance: ", instance)



class ClassicalOptimizer:
    def __init__(self, instance, n, K):

        self.instance = instance
        self.n = n  # number of nodes
        self.K = K  # number of vehicles

    def compute_allowed_combinations(self):
        f = math.factorial
        return f(self.n) / f(self.K) / f(self.n - self.K)

    def cplex_solution(self):

        # refactoring
        instance = self.instance
        n = self.n
        K = self.K

        my_obj = list(instance.reshape(1, n**2)[0]) + [0.0 for x in range(0, n - 1)]
        my_ub = [1 for x in range(0, n**2 + n - 1)]
        my_lb = [0 for x in range(0, n**2)] + [0.1 for x in range(0, n - 1)]
        my_ctype = "".join(["I" for x in range(0, n**2)]) + "".join(
            ["C" for x in range(0, n - 1)]
        )

        my_rhs = (
            2 * ([K] + [1 for x in range(0, n - 1)])
            + [1 - 0.1 for x in range(0, (n - 1) ** 2 - (n - 1))]
            + [0 for x in range(0, n)]
        )
        my_sense = (
            "".join(["E" for x in range(0, 2 * n)])
            + "".join(["L" for x in range(0, (n - 1) ** 2 - (n - 1))])
            + "".join(["E" for x in range(0, n)])
        )

        try:
            my_prob = cplex.Cplex()
            self.populatebyrow(my_prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs)

            my_prob.solve()

        except CplexError as exc:
            print(exc)
            return

        x = my_prob.solution.get_values()
        x = np.array(x)
        cost = my_prob.solution.get_objective_value()

        return x, cost

    def populatebyrow(self, prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs):

        n = self.n

        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)

        prob.set_log_stream(None)
        prob.set_error_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)

        rows = []
        for ii in range(0, n):
            col = [x for x in range(0 + n * ii, n + n * ii)]
            coef = [1 for x in range(0, n)]
            rows.append([col, coef])

        for ii in range(0, n):
            col = [x for x in range(0 + ii, n**2, n)]
            coef = [1 for x in range(0, n)]

            rows.append([col, coef])

        # Sub-tour elimination constraints:
        for ii in range(0, n):
            for jj in range(0, n):
                if (ii != jj) and (ii * jj > 0):

                    col = [ii + (jj * n), n**2 + ii - 1, n**2 + jj - 1]
                    coef = [1, 1, -1]

                    rows.append([col, coef])

        for ii in range(0, n):
            col = [(ii) * (n + 1)]
            coef = [1]
            rows.append([col, coef])

        prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rhs)


# Instantiate the classical optimizer class
classical_optimizer = ClassicalOptimizer(instance, n, K)

# Print number of feasible solutions
print("Number of feasible solutions = " + str(classical_optimizer.compute_allowed_combinations()))

# Solve the problem in a classical fashion via CPLEX
x = None
z = None
try:
    x, classical_cost = classical_optimizer.cplex_solution()
    # Put the solution in the z variable
    z = [x[ii] for ii in range(n**2) if ii // n != ii % n]
    # Print the solution
    print(z)
except:
    print("CPLEX may be missing.")

def extract_route(x, n):
    x_vars = x[:n**2]
    x_matrix = np.reshape(x_vars, (n, n))
    route = []
    current_node = 0  # Start from the depot (node 0)
    visited = set()
    while True:
        route.append(current_node)
        visited.add(current_node)
        # Find the next node
        next_nodes = np.where(x_matrix[current_node] > 0.5)[0]
        # Remove self-loops and already visited nodes
        next_nodes = [node for node in next_nodes if node != current_node and node not in visited]
        if not next_nodes:
            break
        current_node = next_nodes[0]  # Assume only one outgoing arc
    return route


# Visualize the solution
def visualize_solution(xc, yc, x, C, n, K, title_str):
    plt.figure()
    plt.scatter(xc, yc, s=200)
    for i in range(len(xc)):
        plt.annotate(i, (xc[i] + 0.15, yc[i]), size=16, color="r")
    plt.plot(xc[0], yc[0], "r*", ms=20)

    plt.grid()

    for ii in range(0, n**2):

        if x[ii] > 0:
            ix = ii // n
            iy = ii % n
            plt.arrow(
                xc[ix],
                yc[ix],
                xc[iy] - xc[ix],
                yc[iy] - yc[ix],
                length_includes_head=True,
                head_width=0.25,
            )

    plt.title(title_str + " cost = " + str(int(C * 100) / 100.0))
    plt.show()
    # save the plot with the title string
    plt.savefig(title_str + ".png")

if x is not None:
    visualize_solution(xc, yc, x, classical_cost, n, K, "Classical")

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class QuantumOptimizer:
    def __init__(self, instance, n, K):

        self.instance = instance
        self.n = n
        self.K = K

    def binary_representation(self, x_sol=0):

        instance = self.instance
        n = self.n
        K = self.K

        A = np.max(instance) * 100  # A parameter of cost function

        # Determine the weights w
        instance_vec = instance.reshape(n**2)
        w_list = [instance_vec[x] for x in range(n**2) if instance_vec[x] > 0]
        w = np.zeros(n * (n - 1))
        for ii in range(len(w_list)):
            w[ii] = w_list[ii]

        # Some variables I will use
        Id_n = np.eye(n)
        Im_n_1 = np.ones([n - 1, n - 1])
        Iv_n_1 = np.ones(n)
        Iv_n_1[0] = 0
        Iv_n = np.ones(n - 1)
        neg_Iv_n_1 = np.ones(n) - Iv_n_1

        v = np.zeros([n, n * (n - 1)])
        for ii in range(n):
            count = ii - 1
            for jj in range(n * (n - 1)):

                if jj // (n - 1) == ii:
                    count = ii

                if jj // (n - 1) != ii and jj % (n - 1) == count:
                    v[ii][jj] = 1.0

        vn = np.sum(v[1:], axis=0)

        # Q defines the interactions between variables
        Q = A * (np.kron(Id_n, Im_n_1) + np.dot(v.T, v))

        # g defines the contribution from the individual variables
        g = (
            w
            - 2 * A * (np.kron(Iv_n_1, Iv_n) + vn.T)
            - 2 * A * K * (np.kron(neg_Iv_n_1, Iv_n) + v[0].T)
        )

        # c is the constant offset
        c = 2 * A * (n - 1) + 2 * A * (K**2)

        try:
            max(x_sol)
            # Evaluates the cost distance from a binary representation of a path
            fun = (
                lambda x: np.dot(np.around(x), np.dot(Q, np.around(x)))
                + np.dot(g, np.around(x))
                + c
            )
            cost = fun(x_sol)
        except:
            cost = 0

        return Q, g, c, cost

    def construct_problem(self, Q, g, c) -> QuadraticProgram:
        qp = QuadraticProgram()
        for i in range(n * (n - 1)):
            qp.binary_var(str(i))
        qp.objective.quadratic = Q
        qp.objective.linear = g
        qp.objective.constant = c
        return qp

    def solve_problem(self, qp):
        # Set the random seed for reproducibility
        algorithm_globals.random_seed = 10598

        # Determine the number of qubits (binary variables)
        num_qubits = qp.get_num_binary_vars()
        print("Number of qubits used:", num_qubits)

        # Define the ansatz (parameterized quantum circuit)
        ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=1)

        # Transpile the circuit to get depth and gate count
        transpiled_circuit = transpile(ansatz, basis_gates=['u', 'cx'], optimization_level=0)

        circuit_depth = transpiled_circuit.depth()
        gate_count = transpiled_circuit.size()
        print("Circuit depth:", circuit_depth)
        print("Number of gates:", gate_count)
        from qiskit.visualization import circuit_drawer

        circuit_drawer(ansatz, output='mpl', filename='ansatz_circuit.png')

        print(transpiled_circuit)
        

        # Define the optimizer and include a callback to track iterations
        maxiter = 100  # Set the maximum number of iterations
        iteration_counts = []

        def callback(nfev, mean, std):
            print(f"Iteration {nfev}: Mean = {mean}, Std = {std}")
            iteration_counts.append(nfev)

        optimizer = SPSA(maxiter=maxiter, callback=callback)

        # Initialize the VQE algorithm with the sampler, optimizer, and ansatz
        vqe = SamplingVQE(sampler=Sampler(), optimizer=SPSA(), ansatz=ansatz)

        # Create the Minimum Eigen Optimizer using VQE
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=vqe)

        # Solve the problem
        result = min_eigen_optimizer.solve(qp)

        # Compute the cost of the obtained result
        _, _, _, level = self.binary_representation(x_sol=result.x)

        # Get the number of optimizer iterations
        num_iterations = len(iteration_counts)
        print("Number of optimizer iterations:", num_iterations)

        # Measure the runtime (optional)
        return result.x, level


# Instantiate the quantum optimizer class with parameters:
quantum_optimizer = QuantumOptimizer(instance, n, K)

# Check if the binary representation is correct
try:
    if z is not None:
        Q, g, c, binary_cost = quantum_optimizer.binary_representation(x_sol=z)
        print("Binary cost:", binary_cost, "classical cost:", classical_cost)
        if np.abs(binary_cost - classical_cost) < 0.01:
            print("Binary formulation is correct")
        else:
            print("Error in the binary formulation")
    else:
        print("Could not verify the correctness, due to CPLEX solution being unavailable.")
        Q, g, c, binary_cost = quantum_optimizer.binary_representation()
        print("Binary cost:", binary_cost)
except NameError as e:
    print("Warning: Please run the cells above first.")
    print(e)

qp = quantum_optimizer.construct_problem(Q, g, c)

quantum_solution, quantum_cost = quantum_optimizer.solve_problem(qp)

print(quantum_solution, quantum_cost)

# Put the solution in a way that is compatible with the classical variables
x_quantum = np.zeros(n**2)
kk = 0
for ii in range(n**2):
    if ii // n != ii % n:
        x_quantum[ii] = quantum_solution[kk]
        kk += 1

# Function to get grid squares traversed by moving along grid lines
def get_grid_path_squares(x0, y0, x1, y1):
    grid_squares = []
    
    # Round coordinates to nearest grid points
    x0_int, y0_int = int(round(x0)), int(round(y0))
    x1_int, y1_int = int(round(x1)), int(round(y1))
    
    # Horizontal movement from x0 to x1 at y0
    x_range = range(min(x0_int, x1_int), max(x0_int, x1_int) + 1)
    for x in x_range:
        grid_square = (x, y0_int)
        if grid_square not in grid_squares:
            grid_squares.append(grid_square)
    
    # Vertical movement from y0 to y1 at x1
    y_range = range(min(y0_int, y1_int), max(y0_int, y1_int) + 1)
    for y in y_range:
        grid_square = (x1_int, y)
        if grid_square not in grid_squares:
            grid_squares.append(grid_square)
    
    return grid_squares

# Function to get all grid squares traversed in the route
def get_route_grid_squares(coords):
    all_grid_squares = []
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i+1]
        grid_squares = get_grid_path_squares(x0, y0, x1, y1)
        all_grid_squares.extend(grid_squares)
    # Remove duplicates
    all_grid_squares = list(set(all_grid_squares))
    # Sort the grid squares for readability
    return sorted(all_grid_squares, key=lambda x: (x[1], x[0]))  # Sort by y, then x for better readability


def find_shortest_path_between_points(coordinates):
    from collections import deque
    
    def bfs(start, end):
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
                
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
                next_x, next_y = x + dx, y + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < 8 and 0 <= next_y < 8 and 
                    next_pos not in visited):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        return []

    # Connect all points sequentially
    full_path = []
    for i in range(len(coordinates) - 1):
        path_segment = bfs(tuple(coordinates[i]), tuple(coordinates[i + 1]))
        if not path_segment:
            return []
        # Avoid duplicating points between segments
        if full_path:
            path_segment = path_segment[1:]
        full_path.extend(path_segment)
    
    return [list(pos) for pos in full_path]

def print_board_state(board_state, start_pos, road_blocks, goals, path=None):
    """Prints a visual representation of the board state."""
    rows, cols = board_state.shape
    board_visual = [["." for _ in range(cols)] for _ in range(rows)]

    # Mark roadblocks
    for r, c in road_blocks:
        board_visual[r][c] = "R"

    # Mark goals
    for r, c in goals:
        board_visual[r][c] = "G"

    # Mark path
    if path:
        for r, c in path:
            if (r, c) != start_pos and (r, c) not in goals and board_visual[r][c] == ".":
                board_visual[r][c] = "*"

    # Mark start position
    sr, sc = start_pos
    board_visual[sr][sc] = "S"

    # Print the board
    print("\n=== Board State ===")
    for row in board_visual:
        print(" ".join(row))
    print("===================")

def write_path_to_json(path, filename="route.json"):
    import json
    
    with open(filename, 'w') as f:
        json.dump({"path": path}, f, indent=4)

# After solving the problem with CPLEX
if x is not None:
    # Extract the route
    route = extract_route(x, n)
    print("Order in which points were visited (Classical):", route)
    # Get the coordinates in order
    coords = [(xc[i], yc[i]) for i in route]
    print("Coordinates in order visited (Classical):", coords)
    # Calculate grid squares traversed
    #grid_squares = get_route_grid_squares(coords)
    grid_squares = find_shortest_path_between_points(coords)
    print("Grid squares traversed (Classical):", grid_squares)
    # Optionally, you can visualize the grid and path

# After solving the quantum problem
route_quantum = extract_route(x_quantum, n)
print("Order in which points were visited (Quantum):", route_quantum)
# Get the coordinates in order
coords_quantum = [(xc[i], yc[i]) for i in route_quantum]
print("Coordinates in order visited (Quantum):", coords_quantum)
# Calculate grid squares traversed
# grid_squares_quantum = get_route_grid_squares(coords_quantum)
grid_squares_quantum = find_shortest_path_between_points(coords_quantum)
print("Grid squares traversed (Quantum):", grid_squares_quantum)

if len(sys.argv) < 2:
    print("Usage: python3 linear_search.py <config_file>")
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
board_state = np.zeros((board_size, board_size))
start_pos = tuple(config.get("start_pos", [0, 0]))
road_blocks = [tuple(block) for block in config.get("road_blocks", [])]
goals = [tuple(goal) for goal in config.get("goal_pos", [[board_size - 1, board_size - 1]])]
# Print configuration
print("=== Linear Search Configuration ===")
print("Board State Dimensions:", board_state.shape)
print("Start Position:", start_pos)
print("Road Blocks:", road_blocks)
print("Goals:", goals)
# Run linear search
# Print board visualization
path = grid_squares_quantum
print_board_state(board_state, start_pos, road_blocks, goals, grid_squares_quantum)
write_path_to_json(path, "route.json")

