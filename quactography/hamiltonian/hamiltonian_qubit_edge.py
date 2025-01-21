from qiskit.quantum_info import SparsePauliOp
import numpy as np

# Definition of a Hamiltonian class which will contain all informations about the global quantum cost function:


class Hamiltonian_qubit_edge:
    """Creates the Hamiltonian with qubits considered to be edges with the given graph and alpha value"""

    def __init__(self, graph, alpha):
        self.graph = graph
        self.mandatory_c = self.mandatory_cost()
        self.starting_node_c = self.starting_node_cost()
        self.ending_node_c = self.ending_node_cost()
        self.hint_c = self.intermediate_node_cost()
        self.hint_edge_c = self.intermediate_min_edge_cost()
        self.alpha = alpha * self.graph.all_weights_sum / graph.number_of_edges
        self.alpha_d = 8 * self.alpha
        self.alpha_f = 8 * self.alpha
        self.alpha_i = self.alpha
        self.total_hamiltonian = (
            -self.mandatory_c
            + self.alpha_d * (self.starting_node_c) ** 2
            + self.alpha_f * (self.ending_node_c) ** 2
            + self.alpha_i * self.hint_c
            + 0.5 * (self.hint_edge_c)  # ** 2
        ).simplify()

        self.exact_cost, self.exact_path = self.get_exact_sol()

    def mandatory_cost(self):
        """
    Cost term of having a single edge connected to each node of the graph in the given path,
      cost of the weights of the edges taken without penalty.

    Parameters
    ----------
    self : Hamiltonian_qubit_edge object
    Returns
    -------
    SparsePauliOp: Pauli string representing the cost associated with the sum of the weights
      of the edges taken without penalty.
    """

        pauli_weight_first_term = [
            ("I" * self.graph.number_of_edges, self.graph.all_weights_sum / 2)
        ]

        # Z à la bonne position:
        for i in range(self.graph.number_of_edges):
            str1 = (
                "I" * (self.graph.number_of_edges - i - 1) + "Z" + "I" * i,
                -self.graph.weights[0][i] / 2,
            )
            pauli_weight_first_term.append(str1)

        mandatory_cost_h = SparsePauliOp.from_list(pauli_weight_first_term)
        # print(f"\n Cost of given path taken = {mandatory_cost_h}")
        return mandatory_cost_h

    def starting_node_cost(self):
        """
        Cost term of having a single starting node connection (one edge connected to the starting node)

        Parameters
        ----------
        self : Hamiltonian_qubit_edge object
        Returns
        -------
        SparsePauliOp: Pauli string representing the cost associated 
        with the constraint of having a single starting node connection """

        starting_qubit = []
        for node, value in enumerate(self.graph.starting_nodes):
            if value == self.graph.starting_node:
                starting_qubit.append(self.graph.edge_indices[node])
        for node, value in enumerate(self.graph.ending_nodes):
            if value == self.graph.starting_node:
                starting_qubit.append(self.graph.edge_indices[node])
        # print(f"\n Qubit to sum over starting x_i: q({starting_qubit}) - I ")

        pauli_starting_node_term = [
            ("I" * self.graph.number_of_edges, len(starting_qubit) * 0.5 - 1)
        ]

        # Z à la bonne position:
        for _, value in enumerate(starting_qubit):
            str2 = (
                "I" * (self.graph.number_of_edges - (value + 1)) + "Z" + "I" * value,
                -0.5,
            )
            pauli_starting_node_term.append(str2)
        start_node_constraint_cost_h = SparsePauliOp.from_list(pauli_starting_node_term)

        # print(f"\n Start constraint = {start_node_constraint_cost_h}")
        return start_node_constraint_cost_h

    def ending_node_cost(self):
        """
        Cost term of having a single ending node connection (one edge connected to the ending node)

        Parameters
        ----------
        self : Hamiltonian_qubit_edge object
        Returns
        -------
        SparsePauliOp: Pauli string representing the cost associated with 
        the constraint of having a single ending node connection"""
        qubit_end = []
        for node, value in enumerate(self.graph.ending_nodes):
            if value == self.graph.ending_node:
                qubit_end.append(self.graph.edge_indices[node])
        for node, value in enumerate(self.graph.starting_nodes):
            if value == self.graph.ending_node:
                qubit_end.append(self.graph.edge_indices[node])
        # print(f"\nQubit to sum over ending x_i: q({qubit_end}) - I ")

        pauli_end_term = [("I" * self.graph.number_of_edges, len(qubit_end) * 0.5 - 1)]

        # Z at the right place:
        for _, value in enumerate(qubit_end):
            str2 = (
                "I" * (self.graph.number_of_edges - (value + 1)) + "Z" + "I" * value,
                -0.5,
            )
            pauli_end_term.append(str2)
        ending_node_constraint_cost_h = SparsePauliOp.from_list(pauli_end_term)

        # print(f"\n End constraint = {ending_node_constraint_cost_h}")
        return ending_node_constraint_cost_h

    def intermediate_node_cost(self):
        """
        Cost term of having a  pair number of connections to each intermediate node 
        (one edge connected to each intermediate node) 

        Parameters
        ----------
        self : Hamiltonian_qubit_edge object
        Returns
        -------
        SparsePauliOp: Pauli string representing the cost associated with the constraint 
        of having pair number of connections to each intermediate node"""
        # Intermediate connections, constraints:
        int_nodes = []
        for node, value in enumerate(self.graph.starting_nodes):
            if (value != self.graph.starting_node) and (
                value != self.graph.ending_node
            ):
                if value not in int_nodes:
                    int_nodes.append(self.graph.starting_nodes[node])
        for node, value in enumerate(self.graph.ending_nodes):
            if (value != self.graph.starting_node) and (
                value != self.graph.ending_node
            ):
                if value not in int_nodes:
                    int_nodes.append(self.graph.ending_nodes[node])

        # print(f"List of intermediate nodes: {int_nodes} \n")

        liste_qubits_int = [[] for _ in range(len(int_nodes))]
        for i, node in enumerate(int_nodes):
            for node, value in enumerate(self.graph.ending_nodes):
                if value == int_nodes[i]:
                    liste_qubits_int[i].append(self.graph.edge_indices[node])
            for node, value in enumerate(self.graph.starting_nodes):
                if value == int_nodes[i]:
                    liste_qubits_int[i].append(self.graph.edge_indices[node])

        for i in range(len(liste_qubits_int)):
            a = liste_qubits_int[i]
            # print(f"Multiply qubits on intermediate x_i: q({a}) ")

        intermediate_cost_h_term = []
        prod_terms = []
        for list_q in liste_qubits_int:
            prod_term = "I" * self.graph.number_of_edges
            for qubit in list_q:
                prod_term = prod_term[:qubit] + "Z" + prod_term[qubit + 1 :]
            prod_terms.append(prod_term[::-1])

        for i in range(len(liste_qubits_int)):
            intermediate_cost_h_term.append([])

            intermediate_cost_h_term[i] = SparsePauliOp.from_list(
                [("I" * self.graph.number_of_edges, -1.0), (prod_terms[i], 1.0)]
            )

        # print(f"\n Intermediate constraint = {intermediate_cost_h_term}")

        intermediate_cost_h_terms = []
        for i in range(len(intermediate_cost_h_term)):
            intermediate_cost_h_terms.append(intermediate_cost_h_term[i] ** 2)
        # print(f"Sum of intermediate terms squared: {sum(intermediate_cost_h_terms)}")
        return sum(intermediate_cost_h_terms)

    def intermediate_min_edge_cost(self):
        """
        Cost term of penalizing the number of edges connected to each intermediate node 
        (one edge connected to each intermediate node) to avoid loops

        Parameters
        ----------
        self : Hamiltonian_qubit_edge object
        Returns
        -------
        SparsePauliOp: Pauli string representing the cost associated with the constraint of
          having a connection to each intermediate node"""
        # Identify intermediate nodes
        int_nodes = []
        for node, value in enumerate(self.graph.starting_nodes):
            if value != self.graph.starting_node and value != self.graph.ending_node:
                if value not in int_nodes:
                    int_nodes.append(value)
        for node, value in enumerate(self.graph.ending_nodes):
            if value != self.graph.starting_node and value != self.graph.ending_node:
                if value not in int_nodes:
                    int_nodes.append(value)

        liste_qubits_int = []
        for i, node in enumerate(int_nodes):
            for node, value in enumerate(self.graph.ending_nodes):
                if value == int_nodes[i]:
                    liste_qubits_int.append(self.graph.edge_indices[node])
            for node, value in enumerate(self.graph.starting_nodes):
                if value == int_nodes[i]:
                    liste_qubits_int.append(self.graph.edge_indices[node])

        a = liste_qubits_int
        # print(f"Edges present and connected to each int node: q({a}) ")

        pauli_int_edge_term = [("I" * self.graph.number_of_edges, len(int_nodes) * 0.5)]

        # Z at the right place:
        for _, value in enumerate(liste_qubits_int):
            str2 = (
                "I" * (self.graph.number_of_edges - (value + 1)) + "Z" + "I" * value,
                -0.5,
            )
            pauli_int_edge_term.append(str2)
        int_edge_cost_h = SparsePauliOp.from_list(pauli_int_edge_term)
        # print("int edge pauli", int_edge_cost_h)

        # print(f"\n End constraint = {ending_node_constraint_cost_h}")
        return int_edge_cost_h

    def get_exact_sol(self):
        """
        Get the exact solution of the Hamiltonian by diagonalizing it.

        Parameters
        ----------
        self : Hamiltonian_qubit_edge object    
        Returns
        -------
        eigenvalues: list of float
            Eigenvalues of the Hamiltonian, cost of the paths 
        binary_paths: list of binary strings
        Binary paths corresponding to the eigenvectors of the Hamiltonian"""
        mat_hamiltonian = np.array(self.total_hamiltonian.to_matrix())
        eigenvalues, eigenvectors = np.linalg.eig(mat_hamiltonian)

        best_indices = np.where(eigenvalues == np.min(eigenvalues))
        # print(eigenvalues[int("0111", 2)])
        # print("Eigenvalues : ", eigenvalues[best_indices])
        # print("Eigenvectors : ", eigenvectors[best_indices])

        binary_paths = [bin(idx[0]).lstrip("-0b") for idx in best_indices]
        # print("Binary paths : ", binary_paths)

        # costs and paths to all best solutions
        return eigenvalues[best_indices], binary_paths


# # # # TEST:--------------------------------------------------------------------------


# # # mat = np.array([[0, 1, 1, 0], [1, 0, 0, 5], [1, 0, 0, 6], [0, 5, 6, 0]])

# # # # # This is the given format you should use to save the graph, for mat:
# # # # save_graph(mat, np.array([0, 1, 2, 3]), np.array([4, 4]), "rand_graph.npz")
# import sys

# sys.path.append(r"C:\Users\harsh\quactography")

# from quactography.graph.undirected_graph import Graph
# from quactography.adj_matrix.io import load_graph

# # from quactography.hamiltonian.hamiltonian_qubit_node import Hamiltonian_qubit_node
# import numpy as np

# from quactography.adj_matrix.io import save_graph

# my_graph_class = Graph(
#     np.array(
#         [
#             [0, 1, 1, 0],
#             [1, 0, 1, 1],
#             [1, 1, 0, 1],
#             [0, 1, 1, 0],
#         ]
#     ),
#     0,
#     3,
# )

# # # my_graph_class = Graph(
# # #     np.array(
# # #         [
# # #             [0, 1, 1, 1],
# # #             [1, 0, 1, 0],
# # #             [1, 1, 0, 1],
# # #             [1, 0, 1, 0],
# # #         ]
# # #     ),
# # #     1,
# # #     0,
# # # )
# print(my_graph_class.starting_nodes)
# print(my_graph_class.ending_nodes)
# print(my_graph_class.weights)
# print(my_graph_class.edge_indices)

# # Test mandatory_cost
# h = Hamiltonian_qubit_edge(my_graph_class, 1)

# # print(h.mandatory_c)

# # # Test starting_ending_node_cost
# # print(h.starting_node_c)
# # print(h.ending_node_c)

# # Test intermediate_cost
# print(h.hint_edge_c)

# print("total :", h.total_hamiltonian.simplify())
# print(h.exact_cost)
# print(h.exact_path)

# from quactography.hamiltonian.validate import print_hamiltonian_circuit

# print("total")
# print_hamiltonian_circuit(h.total_hamiltonian, ["10101"])
# print("mandatory")
# print_hamiltonian_circuit(h.mandatory_c, ["10101"])
# print("start")
# print_hamiltonian_circuit(h.starting_node_c, ["10101"])
# print("finish")
# print_hamiltonian_circuit(h.ending_node_c, ["10101"])
# print("int")
# print_hamiltonian_circuit(h.hint_c, ["10101"])
# print("INT WE WANT TO TEST")
# print_hamiltonian_circuit(h.hint_c, ["10101"])


# # # print("total2")
# # # print_hamiltonian_circuit(h.total_hamiltonian, ["11111"])
# # # print("mandatory2")
# # # print_hamiltonian_circuit(h.mandatory_c, ["11111"])
# # # print("start2")
# # # print_hamiltonian_circuit(h.starting_node_c, ["11111"])
# # # print("finish2")
# # # print_hamiltonian_circuit(h.ending_node_c, ["11111"])
# # # print("int2")
# # # print_hamiltonian_circuit(h.hint_c, ["11111"])
# # # ------------------------------------------------------------------------------------------------------------------------------------------
