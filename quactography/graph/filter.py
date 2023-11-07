import numpy as np

def remove_orphan_nodes(graph, node_indices, keep_indices=None):
    out_graph = []
    out_it = []
    for it, graph_row in enumerate(graph):
        if np.count_nonzero(graph_row) > 0 or\
            not _test_removable_indice(it, node_indices, keep_indices):
            out_graph.append(graph_row)
            out_it.append(it)
    out_graph = np.take(np.asarray(out_graph), out_it, axis=1)
    out_indices = node_indices[np.asarray(out_it)]

    return out_graph, out_indices


def remove_intermediate_connections(graph, node_indices=None, keep_indices=None):
    skipped_at_least_one = True
    while skipped_at_least_one:
        skipped_at_least_one = False
        for it, graph_row in enumerate(graph):
            if np.count_nonzero(graph_row) == 2 and\
                _test_removable_indice(it, node_indices, keep_indices):
                indices = np.flatnonzero(graph_row)
                graph[indices[0], indices[1]] = np.sum(graph_row)
                graph[indices[1], indices[0]] = np.sum(graph_row)
                graph[it, :] = 0.0
                graph[:, it] = 0.0
                if indices[0] < it and indices[1] < it:
                    skipped_at_least_one = True
    return graph


def _test_removable_indice(it, node_indices, keep_indices):
    if keep_indices is None or node_indices is None:
        return True
    return not (keep_indices == node_indices[it]).any()
