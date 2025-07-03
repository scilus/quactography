#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from quactography.adj_matrix.reconst import (
                    build_adjacency_matrix,
                    build_weighted_graph
)
from quactography.adj_matrix.filter import (
                    remove_orphan_nodes,
                    remove_intermediate_connections,
)
from quactography.image.utils import slice_along_axis
from quactography.adj_matrix.io import save_graph
from scripts.quac_optimal_path_find_max_intensity_diffusion import rap_funct


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
        "--slice_index",
        type=int,
        help="If None, midslice is taken."
        )
    p.add_argument(
        "--axis_name",
        default="axial",
        choices=["sagittal", "coronal", "axial"],
        help="Axis along which a slice is taken.",
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

    nodes_mask = slice_along_axis(
        nodes_mask_im.get_fdata().astype(bool), args.axis_name, args.slice_index
    )

    keep_node_indices = None
    if args.keep_mask:
        keep_mask_im = nib.load(keep_mask)
        keep_mask = slice_along_axis(
            keep_mask_im.get_fdata().astype(bool), args.axis_name, args.slice_index
        )
        keep_node_indices = np.flatnonzero(keep_mask)

    # !!Careful, we remove a dimension, but the SH amplitudes still exist in 3D
    sh = slice_along_axis(sh_im.get_fdata(), args.axis_name, args.slice_index)

    # adjacency graph
    adj_matrix, node_indices = build_adjacency_matrix(nodes_mask)

    # assign edge weights
    weighted_graph, node_indices = build_weighted_graph(
        adj_matrix, node_indices, sh, args.axis_name
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

    if not args.save_only:
        plt.imshow(np.log(weighted_graph + 1))
        plt.show()

    # print("node indices", node_indices)
    # save output
    save_graph(weighted_graph, node_indices, nodes_mask.shape, args.out_graph)
    print("Graph saved")
    path = Path(args.out_graph)

    rap_funct(
        path,
        starting_node=node_indices[0],
        ending_node=node_indices[3],
        output_file="rap_output",
        plt_cost_landscape=False,
        save_only=True
    )
    


if __name__ == "__main__":
    main()
