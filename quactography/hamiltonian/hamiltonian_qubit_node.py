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

        start_node_constraint_cost_h = SparsePauliOp.from_list(pauli_starting_node_term)
        # print(start_node_constraint_cost_h)

        ending_node_constraint_cost_h = SparsePauliOp.from_list(pauli_end_term)
        # print(ending_node_constraint_cost_h)
        return start_node_constraint_cost_h, ending_node_constraint_cost_h

    def intermediate_node_cost(self):
        pass

    #     """Cost term of having an even number of intermediate connections (two edges connected to the intermediate nodes)

    #     Args:
    #         starting_node (int):  Starting node decided by the user
    #         ending_node (int): Ending node decided by the user
    #         starting_nodes (list int): List of nodes in starting_nodesure (according to the adjacency matrix to avoid doublets)
    #         q_indices (list int): Index associated with each qubit according to the adjacency matrix
    #         ending_nodes (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)
    #         number_of_edges (int): Number of edges which is the same as the number of qubits in the graph

    #     Returns:
    #         Sparse pauli op (str): Pauli string representing the cost associated with the constraint of having an even number of intermediate connections
    #     """
    #     # List of ["I" * num_nodes], then replace j element in list by Z (nodes connected to intermediate node k)

    #     initial_int_term = ["I"] * self.graph.num_nodes

    #     # print(self.graph.starting_nodes)
    #     # print(self.graph.ending_nodes)
    #     # print(self.graph.starting_node)
    #     # print(self.graph.ending_node)

    #     # Set an empty dictionary to store the intermediate nodes connected to which other node in the graph:
    #     node_connected = {}
    #     for node, node2 in zip(self.graph.starting_nodes, self.graph.ending_nodes):
    #         if node != self.graph.starting_node and node != self.graph.ending_node:
    #             if node not in node_connected:
    #                 node_connected[node] = [node2]
    #             else:
    #                 node_connected[node].append(node2)
    #         if node2 != self.graph.starting_node and node2 != self.graph.ending_node:
    #             if node2 not in node_connected:
    #                 node_connected[node2] = [node]
    #             else:
    #                 node_connected[node2].append(node)
    #     # Create the right number of terms for every intermediate node:

    #     initial_int_term_list = [initial_int_term] * len(node_connected)
    #     # initial_int_term_list

    #     # Replace the position of the list which are values in the dictionary by Z:
    #     for pos, node_name in enumerate(node_connected):
    #         for node in node_connected[node_name]:
    #             initial_int_term_list[pos] = list(initial_int_term_list[pos])
    #             initial_int_term_list[pos][node] = "Z"
    #             initial_int_term_list[pos] = "".join(initial_int_term_list[pos])
    #             initial_int_term_list[pos] = initial_int_term_list[pos]
    #             # reverse the string to have the correct order of the qubits
    #             initial_int_term_list[pos] = initial_int_term_list[pos][::-1]

    #     # print(initial_int_term_list)

    #     # Now that we have the  product terms, we must add the substraction of the identity operator to each term, elevate each of them to the square, then sum them as a SparsePauliOp:
    #     for i in range(len(initial_int_term_list)):
    #         initial_int_term_list[i] = (initial_int_term_list[i], 1)
    #         initial_int_term_list[i] = [
    #             initial_int_term_list[i],
    #             (("I" * self.graph.num_nodes, -1)),
    #         ]
    #     list_with_identity = initial_int_term_list

    #     # print("identity : ", list_with_identity)

    #     # Create a Pauli Operator with the terms in the list:
    #     for i in range(len(list_with_identity)):
    #         list_with_identity[i] = SparsePauliOp.from_list(list_with_identity[i])

    #     # print("Pauli Operators of list elements : ", list_with_identity)

    #     # Square each term :
    #     for i in range(len(list_with_identity)):
    #         list_with_identity[i] = list_with_identity[i] @ list_with_identity[i]
    #     # print("Squared each term: ", list_with_identity)

    #     # Sum all the terms:
    #     initial_int_term_h = sum(list_with_identity)
    #     sum_intermediate_cost_h_terms = initial_int_term_h

    #     return sum_intermediate_cost_h_terms

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


# Test the Hamiltonian_qubit_node class:

# mat = np.array([[0, 1, 1, 0], [1, 0, 0, 5], [1, 0, 0, 6], [0, 5, 6, 0]])

# # This is the given format you should use to save the graph, for mat:
# save_graph(mat, np.array([0, 1, 2, 3]), np.array([4, 4]), "rand_graph.npz")
import sys

sys.path.append(r"C:\Users\harsh\quactography")

from quactography.graph.undirected_graph import Graph
from quactography.adj_matrix.io import load_graph

# from quactography.hamiltonian.hamiltonian_qubit_node import Hamiltonian_qubit_node
import numpy as np

from quactography.adj_matrix.io import save_graph

my_graph_class = Graph(
    np.array(
        [
            [0, 1, 1, 1, 2],
            [1, 0, 1, 0, 2],
            [1, 1, 0, 1, 2],
            [1, 0, 1, 0, 2],
            [1, 0, 1, 0, 2],
        ]
    ),
    1,
    0,
)
print(my_graph_class.starting_nodes)
print(my_graph_class.ending_nodes)
print(my_graph_class.weights)
print(my_graph_class.q_indices)

# Test mandatory_cost
h = Hamiltonian_qubit_node(my_graph_class, 1)
print(h.mandatory_c)

# Test starting_ending_node_cost
print(h.starting_node_c)
print(h.ending_node_c)

# Test intermediate_cost
print(h.hint_c)

print("total :", h.total_hamiltonian.simplify())
print(h.exact_cost)
print(h.exact_path)
from quactography.hamiltonian.validate import print_hamiltonian_circuit

print("total")
print_hamiltonian_circuit(h.total_hamiltonian, ["11000"])
print("mandatory")
print_hamiltonian_circuit(h.mandatory_c, ["11000"])
print("start")
print_hamiltonian_circuit(h.starting_node_c, ["11000"])
print("finish")
print_hamiltonian_circuit(h.ending_node_c, ["11000"])
print("int")
print_hamiltonian_circuit(h.hint_c, ["11000"])


print("total2")
print_hamiltonian_circuit(h.total_hamiltonian, ["11111"])
print("mandatory2")
print_hamiltonian_circuit(h.mandatory_c, ["11111"])
print("start2")
print_hamiltonian_circuit(h.starting_node_c, ["11111"])
print("finish2")
print_hamiltonian_circuit(h.ending_node_c, ["11111"])
print("int2")
print_hamiltonian_circuit(h.hint_c, ["11111"])
