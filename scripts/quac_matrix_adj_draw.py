#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np

from quactography.adj_matrix.io import load_graph
from quactography.adj_matrix.filter import extract_slice_at_index
import matplotlib.pyplot as plt


"""
Tool to visualize the graph constructed with
diffusion data (white matter mask and fodf peaks).

Input can be either 2D or 3D. When 3D, only a single slice is drawn.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_graph",
                   help="Graph file (npz file)")
    p.add_argument('--slice_index', type=int,
                   help='Slice index to render. None defaults to mid slice.')
    p.add_argument('--axis_name', choices=['sagittal', 'coronal', 'axial'], default='axial',
                   help='Axis to draw. [%(default)s]')
    p.add_argument(
        "--save_only",
        help="Save only the figure without displaying it",
        action="store_true",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, node_indices, vol_dim = load_graph(args.in_graph)
    slice_index = args.slice_index
    if args.slice_index is None:
        if args.axis_name == 'sagittal':
            slice_index = vol_dim[0] // 2
        if args.axis_name == 'coronal':
            slice_index = vol_dim[1] // 2
        if args.axis_name == 'axial':
            slice_index = vol_dim[2] // 2

    x, y, z = np.unravel_index(node_indices, vol_dim)
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    nz = len(np.unique(z))
    if not np.any(np.array([nx, ny, nz]) == 1):  # input graph is 3D
        # Convert input graph in 2D
        weighted_graph, node_indices = extract_slice_at_index(weighted_graph, node_indices, vol_dim,
                                                              slice_index, args.axis_name)
        x, y, z = np.unravel_index(node_indices, vol_dim)
        if args.axis_name == 'sagittal':
            x, y = y, z
        elif args.axis_name == 'coronal':
            x, y = x, z
    else:  # input graph is 2D
        # detect the axis to draw
        if nx == 1:
            x, y = y, z
        elif ny == 1:
            x, y = x, z

    # draw the graph
    weighted_graph = np.triu(weighted_graph)
    for it, node_row in enumerate(weighted_graph):
        nb_adj = np.count_nonzero(node_row)
        if nb_adj > 0:
            w_all = node_row[node_row > 0]
            start_x, start_y = x[it], y[it]
            end_x = x[node_row > 0]
            end_y = y[node_row > 0]
            for vert_id in range(nb_adj):
                w = w_all[vert_id]
                alpha = np.clip(w * 0.9 + 0.1, 0.0, 1.0)
                plt.plot(
                    [start_x, end_x[vert_id]],
                    [start_y, end_y[vert_id]],
                    color="black",
                    alpha=alpha,
                )

    plt.scatter(x, y)
    plt.savefig("graph_adj_mat.png")
    print("Graph saved as graph_adj_mat.png")
    if not args.save_only:
        plt.show()


if __name__ == "__main__":
    main()
