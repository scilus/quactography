from qiskit.quantum_info import SparsePauliOp
import numpy as np

# Definition of a Hamiltonian class which will contain all informations about the global quantum cost function
# when considering the qubits as nodes of a graph:


class Hamiltonian_qubit_node:
    """Creates the Hamiltonian with qubits considered to be nodes with the given graph and alpha value"""

    def __init__(
        self,
        graph,
        alpha,
    ):
        self.graph = graph
        self.mandatory_c = self.mandatory_cost()
        self.starting_node_c = self.starting_ending_node_cost()[0]
        self.ending_node_c = self.starting_ending_node_cost()[1]
        self.hint_c = self.intermediate_node_cost()
        self.alpha = alpha
        self.total_hamiltonian = (
            -self.mandatory_c
            + self.alpha
            * (
                (self.starting_node_c) ** 2 + (self.ending_node_c) ** 2 + self.hint_c
            ).simplify()
        )
        self.exact_cost, self.exact_path = self.get_exact_sol()

    def mandatory_cost(self):
        """Cost of going through a path

        Args:
            num_nodes (int): Number of nodes in the graph
            weights (list int): The weights of the edges
            all_weights_sum (int): Sum of all weights in the graph
            starting_nodes (list int): List of nodes in starting_nodesure (according to the adjacency matrix to avoid doublets)
            ending_nodes (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)

        Returns:
            Sparse pauli op (str):  Pauli string representing the cost of going through a path
        """
        pauli_weight_first_term = [
            ("I" * self.graph.num_nodes, self.graph.all_weights_sum / 4)
        ]

        # Goes trough a list of starting and ending nodes forming all possible edges in the graph, and according to the formula in markdown previous passage,
        # Constructs the Term which includes the cost of the path:
        pos = 0
        for node, node2 in zip(self.graph.starting_nodes, self.graph.ending_nodes):

            str1 = (
                "I" * (self.graph.num_nodes - 1 - node) + "Z" + "I" * node,
                -self.graph.weights[0][pos] / 4,
            )
            str2 = (
                "I" * (self.graph.num_nodes - 1 - node2) + "Z" + "I" * node2,
                -self.graph.weights[0][pos] / 4,
            )
            if node < node2:
                str3 = (
                    "I" * (self.graph.num_nodes - 1 - node2)
                    + "Z"
                    + "I" * (node2 - node - 1)
                    + "Z"
                    + "I" * node,
                    -self.graph.weights[0][pos] / 4,
                )
                pauli_weight_first_term.append(str1)
                pauli_weight_first_term.append(str2)
                pauli_weight_first_term.append(str3)
            pos += 1

        # We must now convert the list of strings containing the Pauli operators to a SparsePauliOp in Qiskit:

        mandatory_cost_h = SparsePauliOp.from_list(pauli_weight_first_term)
        # print(f"\n Cost of given path taken = {mandatory_cost_h}")
        return mandatory_cost_h

    def starting_ending_node_cost(self):
        """Cost term of having only one node connected to the starting node and one node connected to the ending node

        Args:
            num_nodes (int): Number of nodes in the graph
            starting_node (int): Starting node decided by the user
            ending_node (int): Ending node decided by the user
            starting_nodes (list int): List of nodes in starting_nodesure (according to the adjacency matrix to avoid doublets)
            ending_nodes (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)

        Returns:
            Sparse pauli op (list of str): Pauli string representing the cost associated with the constraint of having only one node connected to the starting node and one node connected to the ending node
        """

        departure_nodes = []
        finishing_nodes = []

        # Constructs the Term of the Hamiltonian which makes sure that there is only one node connected to the starting node (only one path taken from begining)
        # Constructs the very similar Term for making sure we also arrive at the ending node with one other intermediate node connected to it:

        # Constant terms for Start constraint:
        pauli_starting_node_term = []

        # Constant terms for End constraint:
        pauli_end_term = []

        for node, node2 in zip(self.graph.starting_nodes, self.graph.ending_nodes):
            start_node = self.graph.starting_node
            end_node = self.graph.ending_node

            if node == start_node:
                departure_nodes.append(node2)
            if node2 == start_node:
                departure_nodes.append(node)
            if node == end_node:
                finishing_nodes.append(node2)
            if node2 == end_node:
                finishing_nodes.append(node)

        for node in departure_nodes:
            str0 = (
                ("I" * self.graph.num_nodes, 0.25),
                (
                    "I" * (self.graph.num_nodes - 1 - self.graph.starting_node)
                    + "Z"
                    + "I" * self.graph.starting_node,
                    -0.25,
                ),
            )
            pauli_starting_node_term.extend(str0)
            if node > self.graph.starting_node:
                str1 = (
                    "I" * (self.graph.num_nodes - 1 - node)
                    + "Z"
                    + "I" * (node - self.graph.starting_node - 1)
                    + "Z"
                    + "I" * self.graph.starting_node,
                    0.25,
                )
                pauli_starting_node_term.append(str1)

            if node < self.graph.starting_node:
                str2 = (
                    "I" * (self.graph.num_nodes - 1 - self.graph.starting_node)
                    + "Z"
                    + "I" * (self.graph.starting_node - node - 1)
                    + "Z"
                    + "I" * node,
                    0.25,
                )
                pauli_starting_node_term.append(str2)

            str3 = (
                "I" * (self.graph.num_nodes - 1 - node) + "Z" + "I" * node,
                -0.25,
            )
            pauli_starting_node_term.append(str3)

        str4 = ("I" * self.graph.num_nodes, -1)

        pauli_starting_node_term.append(str4)

        for node in finishing_nodes:

            if node > self.graph.ending_node:
                str4 = (
                    "I" * (self.graph.num_nodes - 1 - node)
                    + "Z"
                    + "I" * (node - self.graph.ending_node - 1)
                    + "Z"
                    + "I" * self.graph.ending_node,
                    0.25,
                )
                pauli_end_term.append(str4)

            if node < self.graph.ending_node:
                str5 = (
                    "I" * (self.graph.num_nodes - 1 - self.graph.ending_node)
                    + "Z"
                    + "I" * (self.graph.ending_node - node - 1)
                    + "Z"
                    + "I" * node,
                    0.25,
                )
                pauli_end_term.append(str5)

            str6 = (
                "I" * (self.graph.num_nodes - 1 - node) + "Z" + "I" * node,
                -0.25,
            )
            pauli_end_term.append(str6)

            str8 = (
                ("I" * self.graph.num_nodes, 0.25),
                (
                    "I" * (self.graph.num_nodes - 1 - self.graph.ending_node)
                    + "Z"
                    + "I" * self.graph.ending_node,
                    -0.25,
                ),
            )

            pauli_end_term.extend(str8)
        str7 = ("I" * self.graph.num_nodes, -1)
        pauli_end_term.append(str7)
        # print(pauli_starting_node_term)
        # print(pauli_end_term)

        # print("D : ", departure_nodes)
        # print("F : ", finishing_nodes)

        start_node_constraint_cost_h = SparsePauliOp.from_list(
            pauli_starting_node_term * 4
        )
        # print(start_node_constraint_cost_h)

        ending_node_constraint_cost_h = SparsePauliOp.from_list(pauli_end_term * 4)
        # print(ending_node_constraint_cost_h)
        return start_node_constraint_cost_h, ending_node_constraint_cost_h

    def intermediate_node_cost(self):
        """Cost term for having a pair number of nodes connected to each intermediate node

        Args:
            num_nodes (int): Number of nodes in the graph
            starting_node (int): Starting node decided by the graph instance
            ending_node (int): Ending node decided by the graph instance
            starting_nodes (list int): List of nodes in starting_nodesure (according to the adjacency matrix to avoid doublets)
            ending_nodes (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)

        Returns:
            Sparse Pauli op (list of str and weight associated): Return Pauli chain corresponding to the given graph instance (needs to change for each path.... Not done in the code yet)
        """

        # List of intermediate nodes indices:
        intermediate_nodes = []

        # List of list of connexions of each intermediate nodes:
        connected_tos = []

        # print("num nodes", self.graph.num_nodes)

        # Fill the list of intermediate_nodes:
        for i in range(self.graph.num_nodes):

            if i != self.graph.starting_node:
                if i != self.graph.ending_node:
                    intermediate_nodes.append(i)

        # Fill the list of connected_tos with list of connexions for each intermediate node:
        for pos, int_node_index in enumerate(intermediate_nodes):
            connected_to = []

            for node, node2 in zip(self.graph.starting_nodes, self.graph.ending_nodes):
                if node == int_node_index:
                    connected_to.append(node2)
                elif node2 == int_node_index:
                    connected_to.append(node)
            connected_tos.append(connected_to)

        # print(intermediate_nodes)
        # print(connected_tos)

        # List containing the Hamiltonian term for each intermediate term
        list_int_terms = []

        for pos, connexions_list in enumerate(connected_tos):
            init_term = ["I"] * self.graph.num_nodes
            # print(init_term)
            for i in connexions_list:
                init_term[i] = "Z"

            init_term = init_term[::-1]
            init_term = "".join(init_term)
            list_int_terms.extend([((init_term, 1), ("I" * self.graph.num_nodes, -1))])

        # print(list_int_terms)
        # print(len(list_int_terms))

        pauli_op_term_ints = []
        for i in range(len(list_int_terms)):
            pauli_op_term_ints.append(SparsePauliOp.from_list(list_int_terms[i]))

        list_int_verif_terms = []
        pauli_int_verif_terms = []

        # Adding term to verify if the intermediate node is present in the path: Z_int - I for each intermediate node
        for pos, int_node_index in enumerate(intermediate_nodes):
            list_int_verif_terms.append(
                [
                    (
                        "I" * (self.graph.num_nodes - 1 - int_node_index)
                        + "Z"
                        + "I" * int_node_index,
                        1,
                    ),
                    ("I" * self.graph.num_nodes, -1),
                ]
            )
        for i in range(len(intermediate_nodes)):
            pauli_int_verif_terms.append(
                (SparsePauliOp.from_list(list_int_verif_terms[i]))
            )

        # print("int verif", pauli_int_verif_terms)
        # print("pauli int terms", pauli_op_term_ints)
        # Multiply pauli_op_term_ints by pauli_int_verif_terms:
        for i in range(len(pauli_op_term_ints)):
            pauli_op_term_ints[i] = pauli_op_term_ints[i] @ pauli_int_verif_terms[i]

        # Square terms for QUBO method:
        for i in range(len(pauli_op_term_ints)):
            pauli_op_term_ints[i] = pauli_op_term_ints[i] @ pauli_op_term_ints[i]

        sum_intermediate_cost_h_terms = sum(pauli_op_term_ints)
        # print(sum_intermediate_cost_h_terms)
        # print(sum_intermediate_cost_h_terms.simplify())
        return sum_intermediate_cost_h_terms.simplify()  # type: ignore

    def get_exact_sol(self):
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


# """Test the Hamiltonian_qubit_node class: TO REMOVE WHEN TESTING FINISHED---------------------------------------------------------------------------------------------------"""
# # mat = np.array([[0, 1, 1, 0], [1, 0, 0, 5], [1, 0, 0, 6], [0, 5, 6, 0]])

# # # This is the given format you should use to save the graph, for mat:
# # save_graph(mat, np.array([0, 1, 2, 3]), np.array([4, 4]), "rand_graph.npz")
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
#             [0, 1, 1, 1, 1],
#             [1, 0, 1, 0, 1],
#             [1, 1, 0, 1, 1],
#             [1, 0, 1, 0, 1],
#             [1, 0, 1, 0, 1],
#         ]
#     ),
#     1,
#     0,
# )

# # my_graph_class = Graph(
# #     np.array(
# #         [
# #             [0, 1, 1, 1],
# #             [1, 0, 1, 0],
# #             [1, 1, 0, 1],
# #             [1, 0, 1, 0],
# #         ]
# #     ),
# #     1,
# #     0,
# # )
# print(my_graph_class.starting_nodes)
# print(my_graph_class.ending_nodes)
# print(my_graph_class.weights)
# print(my_graph_class.edge_indices)

# # Test mandatory_cost
# h = Hamiltonian_qubit_node(my_graph_class, 1)

# # print(h.mandatory_c)

# # # Test starting_ending_node_cost
# # print(h.starting_node_c)
# # print(h.ending_node_c)

# # Test intermediate_cost
# print(h.hint_c)

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
# # print()

# # print("total2")
# # print_hamiltonian_circuit(h.total_hamiltonian, ["11111"])
# # print("mandatory2")
# # print_hamiltonian_circuit(h.mandatory_c, ["11111"])
# # print("start2")
# # print_hamiltonian_circuit(h.starting_node_c, ["11111"])
# # print("finish2")
# # print_hamiltonian_circuit(h.ending_node_c, ["11111"])
# # print("int2")
# # print_hamiltonian_circuit(h.hint_c, ["11111"])
# # ------------------------------------------------------------------------------------------------------------------------------------------
