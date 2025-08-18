import numpy as np


# Definition of a Graph class which will contain all informations about the graph and its connexions:
class Graph:
    def __init__(self, mat_adj, starting_node, ending_node):
        self.num_nodes = mat_adj.shape[0]
        (
            self.number_of_edges,
            self.weights,
            self.starting_nodes,
            self.ending_nodes,
            self.edge_indices,
            self.all_weights_sum,
            self.max_weight,
        ) = self.connexions_edges(mat_adj, self.num_nodes)
        self.starting_node = starting_node
        self.ending_node = ending_node
        self.min_weight = min(np.tril(mat_adj).flatten())

    def connexions_edges(self, mat_adj, num_nodes):
        """
    Function to determine the connexions between nodes and the edges of the graph, with their weights and indices.

    Parameters
    ----------
    mat_adj : np.array of float
        Adjacency matrix of the graph.
    num_nodes : int
        Number of nodes in the graph.
    Returns
    -------
    number_of_edges : int
        Number of edges in the graph.
    weights : list of float
        List of weights for each edge.
    starting_nodes : list of int
        List of starting nodes for each edge.
    ending_nodes : list of int
        List of ending nodes for each edge.
    edge_indices : list of int
        List of indices for each edge.
    all_weights_sum : float
        Sum of all weights in the graph.
    max_weight : float 
        Maximum weight in the graph
    """
        mat_triang_sup = np.triu(mat_adj)

        # Determine edges indices and weights:
        list_of_nodes_for_naming_edges = []
        ending_nodes = []
        number_of_edges = 0

        for start in range(num_nodes - 1):
            start_node_adj_mat = mat_triang_sup[start]
            end_edge = []
            for node, value in enumerate(start_node_adj_mat):
                if value > 0:
                    end_edge.append(node)
                    number_of_edges += 1
                    ending_nodes.append(node)
            list_of_nodes_for_naming_edges.extend([end_edge])
        #     print(f"Edges from node {start} to node : {end_edge}")
        # print(
        #     f"All possible connexions without doubles: {list_of_nodes_for_naming_edges}\n"
        # )

        edge_indices = []
        starting_nodes = []
        index = 0
        num_nodes_minus_1 = num_nodes - 1
        for i in range(num_nodes_minus_1):
            for _ in list_of_nodes_for_naming_edges[i]:
                starting_nodes.append(i)
                edge_indices.append(index)
                index += 1

        # print(f"Index :{edge_indices}")
        # print(f"Start :{starting_nodes}")
        # print(f"End   :{ending_nodes}")

        weights = []
        for _ in range(number_of_edges):
            for _ in starting_nodes:
                for _ in ending_nodes:
                    weight_qubit = mat_adj[starting_nodes, ending_nodes]
        weights.append(weight_qubit)
        # weights[0][0]
        all_weights_sum = sum(np.tril(mat_adj).flatten())
        max_weight = max(np.tril(mat_adj).flatten())
        return (
            number_of_edges,
            weights,
            starting_nodes,
            ending_nodes,
            edge_indices,
            all_weights_sum,
            max_weight,
        )
