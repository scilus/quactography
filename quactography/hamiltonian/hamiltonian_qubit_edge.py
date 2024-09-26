from qiskit.quantum_info import SparsePauliOp
import numpy as np

# Definition of a Hamiltonian class which will contain all informations about the global quantum cost function:


class Hamiltonian:
    """Creates the Hamiltonian with qubits considered to be edges with the given graph and alpha value"""

    def __init__(
        self,
        graph,
        alpha,
    ):
        self.graph = graph
        self.mandatory_c = self.mandatory_cost()
        self.starting_node_c = self.starting_node_cost()
        self.ending_node_c = self.ending_node_cost()
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
            number_of_edges (int): Number of edges in the graph
            weights (list int): The weights of the edges
            all_weights_sum (int): Sum of all weights in the graph

        Returns:
            Sparse pauli op (str):  Pauli string representing the cost of going through a path
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
        """Cost term of having a single starting_nodesure connection (one edge connected to the starting node)

        Args:
            starting_node (int): Starting node decided by the user
            starting_nodes (list int):  List of nodes in starting_nodes (according to the adjacency matrix to avoid doublets)
            q_indices (list int): Index associated with each qubit according to the adjacency matrix
            ending_nodes (list int):  List of nodes in end (according to the adjacency matrix to avoid doublets)
            number_of_edges (int): Number of edges which is the same as the number of qubits in the graph

        Returns:
            Sparse pauli op (str): Pauli string representing the cost associated with the constraint of having a single starting_nodesure connection
        """

        starting_qubit = []
        for node, value in enumerate(self.graph.starting_nodes):
            if value == self.graph.starting_node:
                starting_qubit.append(self.graph.q_indices[node])
        for node, value in enumerate(self.graph.ending_nodes):
            if value == self.graph.starting_node:
                starting_qubit.append(self.graph.q_indices[node])
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
        """Cost term of having a single end connection (one edge connected to the ending node)

        Args:
            ending_node (int): Ending node decided by the user
            starting_nodes (list int): List of nodes in starting_nodes (according to the adjacency matrix to avoid doublets)
            q_indices (list int): Index associated with each qubit according to the adjacency matrix
            ending_nodes (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)
            number_of_edges (int): Number of edges which is the same as the number of qubits in the graph

        Returns:
        Sparse pauli op (str): Pauli string representing the cost associated with the constraint of having a single end connection
        """
        qubit_end = []
        for node, value in enumerate(self.graph.ending_nodes):
            if value == self.graph.ending_node:
                qubit_end.append(self.graph.q_indices[node])
        for node, value in enumerate(self.graph.starting_nodes):
            if value == self.graph.ending_node:
                qubit_end.append(self.graph.q_indices[node])
        # print(f"\nQubit to sum over ending x_i: q({qubit_end}) - I ")

        pauli_end_term = [("I" * self.graph.number_of_edges, len(qubit_end) * 0.5 - 1)]

        # Z à la bonne position:
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
        """Cost term of having an even number of intermediate connections (two edges connected to the intermediate nodes)

        Args:
            starting_node (int):  Starting node decided by the user
            ending_node (int): Ending node decided by the user
            starting_nodes (list int): List of nodes in starting_nodesure (according to the adjacency matrix to avoid doublets)
            q_indices (list int): Index associated with each qubit according to the adjacency matrix
            ending_nodes (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)
            number_of_edges (int): Number of edges which is the same as the number of qubits in the graph

        Returns:
            Sparse pauli op (str): Pauli string representing the cost associated with the constraint of having an even number of intermediate connections
        """
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
                    liste_qubits_int[i].append(self.graph.q_indices[node])
            for node, value in enumerate(self.graph.starting_nodes):
                if value == int_nodes[i]:
                    liste_qubits_int[i].append(self.graph.q_indices[node])

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
