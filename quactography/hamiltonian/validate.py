import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator


def print_hamiltonian_circuit(hamiltonian_term, binary_paths_classical_read):
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    circuit = QuantumCircuit(len(binary_paths_classical_read[0]))
    for i in range(len(binary_paths_classical_read)):
        for j in range(len(binary_paths_classical_read[i])):
            if binary_paths_classical_read[i][j] == "1":
                circuit.x(j)

        print(
            # circuit,
            "\n Cost for path (classical read -> left=q0)",
            binary_paths_classical_read[i],
            " : ",
            estimator.run(circuit, hamiltonian_term).result().values[0],
        )


# # Code to test Hamiltonian class and its methods for a given graph and qubit node:
# import numpy as np
# import sys

# sys.path.append(r"C:\Users\harsh\quactography")

# from quactography.graph.undirected_graph import Graph
# from quactography.adj_matrix.io import load_graph
# from quactography.hamiltonian.hamiltonian_qubit_node import Hamiltonian
# my_graph = load_graph(
#     r"C:\Users\harsh\quactography\quactography\hamiltonian\rand_graph.npz"
# )
# my_graph_class = Graph(my_graph[0], 1, 0)
# # Test mandatory_cost
# h = Hamiltonian(my_graph_class, 1)
# print(h.mandatory_c)
