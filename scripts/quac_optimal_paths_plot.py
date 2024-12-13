#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from quactography.visu.optimal_paths_edge import visualize_optimal_paths_edge


"""
Tool to visualize and plot the optimal path (most probable) on a graph. 
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_mat_adj", 
                   help="Adjacency matrix (npz file)", type=str)
    p.add_argument(
        "in_opt_res",
        nargs="+",
        help="List of input files to plot optimal paths (npz files)",
    )
    p.add_argument(
        "out_visu_path",
        help="Output file name for visualisation (png image)",
        type=str,
    )
    p.add_argument(
        "--save_only",
        help="Save only the figure without displaying it",
        action="store_true",
    )

    return p


def main():
    """
    Plots of the resulting paths found and information on wether or not it is the right path that has been found or not.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    for i in range(len(args.in_opt_res)):
        visualize_optimal_paths_edge(
            args.in_mat_adj,
            args.in_opt_res[i],
            args.out_visu_path + " " + str(i),
            args.save_only
        )


if __name__ == "__main__":
    main()
