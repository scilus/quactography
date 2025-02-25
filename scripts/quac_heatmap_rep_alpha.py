#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from quactography.visu.optimal_path_odds import visu_heatmap

"""
Tool to visualize and plot the optimal path (most probable) on a graph.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "in_opt_res",
        help="Directory of input files to plot optimal paths (npz files)",
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
    parser = _build_arg_parser()
    args = parser.parse_args()

    visu_heatmap(
        args.in_opt_res,
        args.out_visu_path,
        args.save_only,
    )


if __name__ == "__main__":
    main()
