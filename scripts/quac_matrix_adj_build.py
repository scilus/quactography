#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from quactography.adj_matrix.reconst import (
                    build_adjacency_matrix,
                    build_weighted_graph
)
from quactography.adj_matrix.filter import (
                    remove_orphan_nodes,
                    remove_intermediate_connections,
                    extract_slice_at_index
)
from quactography.adj_matrix.io import save_graph


"""
Tool to build adjacency matrix from diffusion data (white matter mask and fodf peaks)
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_nodes_mask",
                   help="Input nodes mask image (.nii.gz file)")
    p.add_argument("in_sh",
                   help="Input SH image. (nii.gz file)")
    p.add_argument("out_graph",
                   help="Output graph file name (npz file)")

    p.add_argument(
        "--keep_mask",
        help="Nodes that must not be filtered out."
        )
    p.add_argument(
        "--threshold",
        default=0.0,
        type=float,
        help="Cut all weights below a given threshold. [%(default)s]",
    )
    p.add_argument(
        '--sh_order',
        type=int,
        default=12,
        help='Maximum SH order. [%(default)s]'
    )
    p.add_argument(
        "--slice_index",
        type=int,
        help="If None, a 3D graph is built."
        )
    p.add_argument(
        "--axis_name",
        default="axial",
        choices=["sagittal", "coronal", "axial"],
        help="Axis along which a slice is taken. Ignored when slice_index is None.",
    )
    p.add_argument(
        "--save_only",
        action="store_true",
        help="Does not plot the matrix, only saves a copy of npz adjacency matrix file"
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    nodes_mask_im = nib.load(args.in_nodes_mask)
    sh_im = nib.load(args.in_sh)

    nodes_mask = nodes_mask_im.get_fdata().astype(bool)

    keep_node_indices = None
    if args.keep_mask:
        keep_mask = nib.load(args.keep_mask).get_fdata().astype(bool)
        keep_node_indices = np.flatnonzero(keep_mask)

    sh = sh_im.get_fdata()

    # adjacency graph
    adj_matrix, node_indices = build_adjacency_matrix(nodes_mask)

    # assign edge weights
    weighted_graph, node_indices = build_weighted_graph(
        adj_matrix, node_indices, sh, args.sh_order
    )

    # # Could be added in the code if needed:
    # # Select sub-graph and filter:___________________________________________
    # select_1 = 30
    # select_2 = 200

    # weighted_graph = weighted_graph[select_1:select_2, select_1:select_2]
    # # print(weighted_graph)
    # # _______________________________________________________________________

    # filter graph edges by weight
    weighted_graph[weighted_graph < args.threshold] = 0.0

    # remove intermediate nodes that connect only two nodes
    weighted_graph = remove_intermediate_connections(
        weighted_graph, node_indices, keep_node_indices
    )

    # remove nodes without edges
    weighted_graph, node_indices = remove_orphan_nodes(
        weighted_graph, node_indices, keep_node_indices
    )

    if args.slice_index is not None:
        weighted_graph, node_indices = extract_slice_at_index(
            weighted_graph, node_indices, nodes_mask.shape, args.slice_index, args.axis_name
        )

    if not args.save_only:
        plt.imshow(np.log(weighted_graph + 1))
        plt.show()

    # save output
    save_graph(weighted_graph, node_indices, nodes_mask.shape, args.out_graph)
    print("Graph saved")


if __name__ == "__main__":
    main()
