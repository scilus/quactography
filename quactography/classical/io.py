import numpy as np
import json
import networkx as nx


def save_graph(G, output_base, copies=1):
    """
    Save a graph to .npz files as arrays of nodes and edges.

    Parameters
    ----------
    G : networkx.Graph
        The graph to be saved.
    output_base : str
        The base name for output files (e.g., 'graph' will become 'graph_0.npz').
    copies : int, optional
        The number of copies to save. Default is 1.

    Returns
    -------
    None
    """
    for i in range(copies):
        output_file = f"{output_base}_{i}.npz"
        nodes = list(G.nodes())
        edges = list(G.edges())

        nodes_array = np.array(nodes)
        edges_array = np.array(edges)

        np.savez(output_file, nodes=nodes_array, edges=edges_array)
        print(f"Copie {i + 1}/{copies} saved as '{output_file}'.")


def load_the_graph(file_path):
    """
    Load a graph from a .json or .npz file.

    Parameters
    ----------
    file_path : str
        Path to the graph file. Supported formats: .json, .npz.

    Returns
    -------
    networkx.Graph
        The loaded graph.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    if file_path.endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        G = nx.Graph()
        for node in data["nodes"]:
            G.add_node(tuple(node))

        for edge in data["edges"]:
            node1, node2 = tuple(map(tuple, edge[:2]))
            weight = edge[2] if len(edge) > 2 else 1
            G.add_edge(node1, node2, weight=weight)

        return G

    if file_path.endswith(".npz"):
        data = np.load(file_path, allow_pickle=True)
        adjacency_matrix = data["adjacency_matrix"]
        raw_nodes = data["node_indices"]

        node_indices = [
            tuple(node) if isinstance(node, (list, np.ndarray, tuple)) else (node,)
            for node in raw_nodes
        ]
        node_indices = [(n[0], 0) if len(n) == 1 else n for n in node_indices]

        print(
            f"Loaded adjacency matrix of shape {adjacency_matrix.shape} "
            f"with {len(node_indices)} nodes."
        )

        G = nx.Graph()
        for node in node_indices:
            G.add_node(node)

        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix[i])):
                if adjacency_matrix[i, j] > 0:
                    G.add_edge(
                        node_indices[i], node_indices[j],
                        weight=adjacency_matrix[i, j]
                    )

        print(f"Nodes in G: {list(G.nodes())}")
        print(
            f"Graph charged with {G.number_of_nodes()} nodes and "
            f"{G.number_of_edges()} edges."
        )
        return G

    raise ValueError("Unsupported file format. Use either .json or .npz.")
