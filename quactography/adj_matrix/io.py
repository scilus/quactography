import numpy as np


def save_graph(adjacency_matrix, node_indices, vol_dims, outfile):
    np.savez(outfile,
             adjacency_matrix=adjacency_matrix,
             node_indices=node_indices,
             vol_dims=vol_dims)


def load_graph(in_file):
    npzfile = np.load(in_file)
    return (npzfile['adjacency_matrix'],
            npzfile['node_indices'], npzfile['vol_dims'])
