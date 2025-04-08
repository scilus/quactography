#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from quactography.visu.gs_square_loss_for_p import *
from quactography.visu.optimal_path_odds import *


"""
Tool to visualize and plot the optimal path (most probable) on a graph.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "in_opt_res",
        help="Directory of input files to plot squareloss for"
            "(npz files)",
    )
    p.add_argument(
        "out_visu_path",
        help="Output file name for visualisation (png image)",
        type=str,
    )
    p.add_argument(
        "comparator",
        choices=["rep", "alpha"],
        help="Plot the square loss for either repetitions "
            "or alphas parameter [rep,alpha]",
        type=str,
    )
    p.add_argument(
        "--save_only",
        help="Save only the figure without displaying it",
        action="store_true",
    )

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.comparator == "alpha":
        visualize_optimal_paths_edge_alpha(
                args.in_opt_res,
                args.out_visu_path,
                args.save_only
            )
        visualize_optimal_prob_alpha(
                args.in_opt_res,
                args.out_visu_path,
                args.save_only
        )

    else:
        visualize_optimal_paths_edge_rep(
                args.in_opt_res,
                args.out_visu_path,
                args.save_only
        )
        visualize_optimal_prob_rep(
                args.in_opt_res,
                args.out_visu_path,
                args.save_only
        )


if __name__ == "__main__":
    main()
