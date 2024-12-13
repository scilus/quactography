import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from quactography.adj_matrix.reconst import build_adjacency_matrix, build_weighted_graph
from quactography.adj_matrix.filter import (
    remove_orphan_nodes,
    remove_intermediate_connections,
)
from quactography.image.utils import slice_along_axis
from quactography.adj_matrix.io import save_graph

    """Tool to build adjacency matrix from diffusion data (white matter mask and fodf peaks)
    """
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_nodes_mask", help="Input nodes mask image")
    p.add_argument("in_sh", help="Input SH image.")
    p.add_argument("out_graph", help="Output graph file name.")

    p.add_argument("--keep_mask", help="Nodes that must not be filtered out.")

    p.add_argument(
        "--threshold",
        default=0.0,
        type=float,
        help="Cut all weights below a given threshold. [%(default)s]",
    )
    p.add_argument("--slice_index", type=int, help="If None, midslice is taken.")
    p.add_argument(
        "--axis_name",
        default="axial",
        choices=["sagittal", "coronal", "axial"],
        help="Axis along which a slice is taken.",
    )
    return p


def main():
    """
    Build graph from SH image.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    nodes_mask_im = nib.load(args.in_nodes_mask)  # type: ignore
    sh_im = nib.load(args.in_sh)  # type: ignore

    nodes_mask = slice_along_axis(
        nodes_mask_im.get_fdata().astype(bool), args.axis_name, args.slice_index  # type: ignore
    )

    keep_node_indices = None
    if args.keep_mask:
        keep_mask_im = nib.load(args.keep_mask)  # type: ignore
        keep_mask = slice_along_axis(
            keep_mask_im.get_fdata().astype(bool), args.axis_name, args.slice_index  # type: ignore
        )
        keep_node_indices = np.flatnonzero(keep_mask)

    # !! Careful, we remove a dimension, but the SH amplitudes still exist in 3D
    sh = slice_along_axis(sh_im.get_fdata(), args.axis_name, args.slice_index)  # type: ignore

    # adjacency graph
    adj_matrix, node_indices = build_adjacency_matrix(nodes_mask)

    # assign edge weights
    weighted_graph, node_indices = build_weighted_graph(
        adj_matrix, node_indices, sh, args.axis_name
    )

    # # Could be added in the code if needed:
    # # Select sub-graph and filter:__________________________________________________
    # select_1 = 30
    # select_2 = 200

    # weighted_graph = weighted_graph[select_1:select_2, select_1:select_2]
    # # print(weighted_graph)
    # # ________________________________________________________________________________

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
    plt.imshow(np.log(weighted_graph + 1))
    plt.show()
    # print("node indices", node_indices)
    # save output
    save_graph(weighted_graph, node_indices, nodes_mask.shape, args.out_graph)


if __name__ == "__main__":
    main()
