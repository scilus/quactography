#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from quactography.visu.dist_prob_edge import plot_distribution_comparison


"""
Tool to visualize 10% of most optimal paths found by QAOA
from multiple simulation for comparison. User responsibility
to make sure data file are for the same graph. 
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "in_opt_res",
        nargs="+",
        help="List of input files to plot distribution",
    )
    p.add_argument(
        "visual_dist_output_file_selected",
        help="Output file name for visualisation",
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

    plot_distribution_comparison(
        args.in_opt_res,
        args.visual_dist_output_file_selected,
        args.save_only,
    )


if __name__ == "__main__":
    main()
