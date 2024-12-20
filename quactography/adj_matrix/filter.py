import numpy as np


# Filter out nodes that aren't connected to any other:
def remove_orphan_nodes(graph, node_indices, keep_indices=None):
    """
    Remove nodes that are not connected to any other node.

    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph.
    node_indices : np.ndarray
        Indices of the nodes in the graph.
    keep_indices : np.ndarray, optional
        Indices of the nodes to keep in the graph. The default is None.
    Returns
    -------
    out_graph : np.ndarray
        Adjacency matrix of the graph with orphan nodes removed.
    out_indices : np.ndarray
        Indices of the nodes in the graph with orphan nodes removed.
    """
    out_graph = []
    out_it = []
    for it, graph_row in enumerate(graph):
        if np.count_nonzero(graph_row) > 0 or not _test_removable_indice(
            it, node_indices, keep_indices
        ):
            out_graph.append(graph_row)
            out_it.append(it)
    out_graph = np.take(np.asarray(out_graph), out_it, axis=1)
    out_indices = node_indices[np.asarray(out_it)]

    return out_graph, out_indices


# Remove nodes that do not add a change in direction between the two nodes 
# it is connected to where the direction stays the same:
def remove_intermediate_connections(graph, node_indices=None, keep_indices=None):
    """
    Remove nodes that do not add a change in direction between the two nodes it is connected to where the direction stays the same.

    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph.
    node_indices : np.ndarray, optional
        Indices of the nodes in the graph. The default is None.
    keep_indices : np.ndarray, optional
        Indices of the nodes to keep in the graph. The default is None.
    Returns
    -------
    graph : np.ndarray
        Adjacency matrix of the graph with intermediate nodes removed.  
    """
    skipped_at_least_one = True
    while skipped_at_least_one:
        skipped_at_least_one = False
        for it, graph_row in enumerate(graph):
            if np.count_nonzero(graph_row) == 2 and _test_removable_indice(
                it, node_indices, keep_indices
            ):
                indices = np.flatnonzero(graph_row)
                # Here we sum the weights:
                graph[indices[0], indices[1]] = np.sum(graph_row)
                graph[indices[1], indices[0]] = np.sum(graph_row)
                # We replace with zero the node that is taken out:
                graph[it, :] = 0.0
                graph[:, it] = 0.0
                if indices[0] < it and indices[1] < it:
                    skipped_at_least_one = True
    return graph


# Remove nodes that do not add a change in direction between the
# two nodes it is connected to where the direction stays the same
# by multiplying instead of adding the weights:
def remove_intermediate_connections_prod_instead_sum(
    graph, node_indices=None, keep_indices=None
):
    """
    Remove nodes that do not add a change in direction between the two nodes it is connected to where the direction stays the same by multiplying instead of adding the weights.

    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph.
    node_indices : np.ndarray, optional
        Indices of the nodes in the graph. The default is None.
    keep_indices : np.ndarray, optional
        Indices of the nodes to keep in the graph. The default is None.
    Returns
    -------
    graph : np.ndarray
        Adjacency matrix of the graph with intermediate nodes removed.
    """
    skipped_at_least_one = True
    while skipped_at_least_one:
        skipped_at_least_one = False
        for it, graph_row in enumerate(graph):
            if np.count_nonzero(graph_row) == 2 and _test_removable_indice(
                it, node_indices, keep_indices
            ):
                indices = np.flatnonzero(graph_row)
                # Here we sum the weights:

                graph[indices[0], indices[1]] = np.prod(graph_row)
                graph[indices[1], indices[0]] = np.prod(graph_row)
                # We replace with zero the node that is taken out:
                graph[it, :] = 0.0
                graph[:, it] = 0.0
                if indices[0] < it and indices[1] < it:
                    skipped_at_least_one = True
    return graph


def _test_removable_indice(it, node_indices, keep_indices):
    if keep_indices is None or node_indices is None:
        return True
    return not (keep_indices == node_indices[it]).any()


# Remove all-zero columns and rows from the adjacency matrix:
def remove_zero_columns_rows(mat: np.ndarray):
    """
    Remove all-zero columns and rows from the adjacency matrix.

    Parameters
    ----------
    mat : np.ndarray
        Adjacency matrix of the graph
    Returns
    -------
    np.ndarray
        Adjacency matrix of the graph with all-zero columns and rows removed.
    """
    zero_cols = np.all(mat == 0, axis=0)  # type: ignore
    zero_rows = np.all(mat == 0, axis=1)
    non_zero_cols = np.where(~zero_cols)[0]
    non_zero_rows = np.where(~zero_rows)[0]
    return mat[np.ix_(non_zero_rows, non_zero_cols)]
