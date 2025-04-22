import numpy as np

def save_graph(G, output_base, copies=1):
    """
    Save the structure of a 2D grid graph into compressed `.npz` files.

    Parameters
    ----------
    G : networkx.Graph
        A 2D grid graph where each node is represented as a tuple `(x, y)`.
        The graph typically connects nodes to their 4-neighbors (up, down, left, right).
        Nodes corresponding to obstacles should already be removed from the graph.
    output_base : str
        Base name for the output files. Each saved file will be named as
        "<output_base>_<index>.npz".
    copies : int, optional (default=1)
        Number of identical copies of the graph to save.

    Returns
    -------
    None
        Files are saved to disk in NumPy's compressed `.npz` format, containing:
        - 'nodes': array of node coordinates
        - 'edges': array of edge pairs (as tuples of coordinates)

    Example
    -------
    >>> save_graph(G, "grid_graph", copies=3)
    Saves: grid_graph_0.npz, grid_graph_1.npz, grid_graph_2.npz
    """
    for i in range(copies):
        output_file = f"{output_base}_{i}.npz"
        nodes = list(G.nodes())
        edges = list(G.edges())

        nodes_array = np.array(nodes)
        edges_array = np.array(edges)

        np.savez(output_file, nodes=nodes_array, edges=edges_array)
        print(f"Copy {i+1}/{copies} saved as '{output_file}'.")
