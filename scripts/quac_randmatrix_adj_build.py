#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse

from quactography.adj_matrix.filter import remove_zero_columns_rows
from quactography.adj_matrix.io import save_graph


"""
Tool to build random matrix with specified number of nodes or edges, work in progress (WIP) instead of diffusion data. 
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("num_nodes", 
                   help="Number of nodes desired in the graph.", 
                   type=int)
    p.add_argument("num_edges", 
                   help="Number of edges desired in the graph.", 
                   type=int)
    p.add_argument(
        "edges_matter",
        help="If True, num_edges is the exact number of edges in the graph, if False, num_edges is the maximum number of edges in the graph.",
        type=bool,
    )
    p.add_argument("out_graph", 
                   help="Output graph file name (npz file)", 
                   type=str)

    return p


# New method to generate random adjacency matrix:
def main():
    """Generate a random adjacency matrix given number of nodes and edges.

    Args:
        num_nodes (int): number of nodes desired in the graph
        num_edges (int): number of edges desired in the graph
        edges_matter (bool): If False, num_edges is the maximum number of edges in the graph,  
        if True , num_edges is the exact number of edges in the graph,

    Returns:
        mat_adj (np.ndarray) : adjacency matrix of the graph
        adj_matrix_from_csv (pd.DataFrame) : adjacency matrix of the graph in pandas DataFrame format
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    max_num_edges = ((args.num_nodes * args.num_nodes) - args.num_nodes) / 2
    # print(f"max number of edges: {max_num_edges}")

    if args.edges_matter == True:
        args.num_nodes = int((1 + np.sqrt(1 + 8 * args.num_edges)) / 2)
    # print(args.num_nodes)
    # print(args.num_edges)

    if args.edges_matter == False:
        if args.num_edges > max_num_edges:
            args.num_edges = max_num_edges  # type: ignore

    # print("num edges wanted", args.num_edges)
    mat = np.zeros((args.num_nodes, args.num_nodes), dtype=float)
    for i in range(args.num_nodes):
        for j in range(args.num_nodes):
            if i is not j:
                mat[i, j] = np.random.randint(1, 3 + 1)
                mat[j, i] = mat[i, j]

    num_edges_in_mat = 0
    mat_flatten = mat.flatten()
    for i in mat_flatten:
        if i != 0:
            num_edges_in_mat += 1
    num_edges_in_mat = num_edges_in_mat / 2
    num_edges_too_much = num_edges_in_mat - args.num_edges
    # print("num edges to delete:", num_edges_too_much)

    # print(mat_flatten)
    mat_triu = np.triu(mat)
    mat_triu_flatten = mat_triu.flatten()
    # print(num_edges_too_much)
    if num_edges_too_much > 0:
        while num_edges_too_much > 0:
            for pos, value in enumerate(mat_triu_flatten):
                if value != 0:
                    first_non_zero_pos_upper_mat = pos
                    break

            # print(first_non_zero_pos_upper_mat)
            mat_flatten[first_non_zero_pos_upper_mat] = 0
            mat_triu_flatten[first_non_zero_pos_upper_mat] = 0
            # print(mat_flatten)
            mat = mat_flatten.reshape((args.num_nodes, args.num_nodes))

            for i in range(args.num_nodes):
                for j in range(args.num_nodes):
                    if i is not j:
                        mat[j, i] = mat[i, j]
            # print(num_edges_too_much)
            num_edges_too_much -= 1

    # print(num_edges_too_much)
    # print(mat_triu_flatten)

    # Remove all-zero columns and rows
    mat = remove_zero_columns_rows(mat)

    # Save the graph:
    save_graph(mat, np.arange(mat.shape[0]), mat.shape, args.out_graph)
    print("Graph saved in", args.out_graph)


if __name__ == "__main__":

    main()
