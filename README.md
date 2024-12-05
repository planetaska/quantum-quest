# Quantum Quest

The repository for a simple quantum algorithm game.

### Main concept

- Applying what we have learned in the classroom to practical exercises
- Expand on grid-world by creating a version of the robot that operates based on quantum principles
- Provide a few different vacuum cleaner robot models, each differentiated by its search algorithm

### Implemented algorithms

- Linear search (classical)
- Dijkstra algorithm (classical)
- VQE algorithm (quantum)
- Grover's algorithm (quantum)
- Quantum Walk algorithm (hybrid)
- Grover's algorithm (hybrid)

### System Dependencies

- Godot 4 must be installed
- Python 3.13 or later
- Qiskit 1.2 or later
- qiskit_aer
- qiskit_algorithms
- qiskit_ibm_runtime

### Running the game

In Godot 4, open the project and hit the Run Project button.

### Running the algorithms

Each algorithms in the `/algorithms` folder can be run individually.

For example:

`python3 vqe.py config.json`

Will execute the VQE algorithm. Make sure you have Qiskit installed in your virtual environment.