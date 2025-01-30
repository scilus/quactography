#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import ast
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
        help="If True, num_edges is the exact number of edges in the graph,"
          "if False, num_edges is the maximum number of edges in the graph.",
        type=ast.literal_eval,
    )
    p.add_argument("out_graph", 
                   help="Output graph file name (npz file)", 
                   type=str)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    num_nodes = args.num_nodes
    num_edges = args.num_edges
    # print(f"max number of edges: {max_num_edges}")

    if args.edges_matter:
        num_nodes = int(np.ceil((1 + np.sqrt(1 + 8 * num_edges)) / 2))
    # print(args.num_nodes)
    # print(args.num_edges)
    else:
        num_edges = ((num_nodes * num_nodes) - num_nodes) / 2  # type: ignore

    # print("num edges wanted", args.num_edges)
    mat = np.zeros((num_nodes, num_nodes), dtype=float)
    # used in a situation where the maximum number of edges in a matrix is more than desired
    num_edges_too_much = (((num_nodes * num_nodes) - num_nodes) / 2) - num_edges 
    


    if num_edges_too_much > 0:
        while num_edges_too_much > 0:
            for i in range(1,num_nodes):
                for j in range(i):
                    if(np.random.randint(0, 10)  > 2 and num_edges_too_much > 0):
                        mat[i, j] = 0
                        mat[j, i] = mat[i, j]
                        num_edges_too_much -= 1
                    else:
                        mat[i, j] = np.random.randint(1, 3 + 1)
                        mat[j, i] = mat[i, j]
    else:
        for i in range(num_nodes):
            for j in range(i):
                mat[i, j] = np.random.randint(1, 3 + 1)
                mat[j, i] = mat[i, j]

    # print("num edges to delete:", num_edges_too_much)

    # Remove all-zero columns and rows
    mat = remove_zero_columns_rows(mat)

    # Save the graph:
    save_graph(mat, np.arange(mat.shape[0]), mat.shape, args.out_graph)
    print("Graph saved in", args.out_graph)


if __name__ == "__main__":

    main()
