#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate random grids/graphs with obstacles.
This script can optionally suppress the output
display of the grid and graph nodes.
The generated graphs are saved as .npz files.
"""

import argparse
import sys

from quactography.classical.utils.random_grid_generator import generate_grid, save_graph


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--size', type=int, default=10,
        help="Size of the grid (the grid will be of shape size x size)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--ratio', type=float,
        help="Ratio of obstacles (e.g., 0.2 = 20%)."
    )
    group.add_argument(
        '--number', type=int,
        help="Exact number of obstacles."
    )

    parser.add_argument(
        '--output', required=True,
        help="Output format: 'filename.npz;<number>'. "
             "Generates <number> files like 'filename_0.npz', etc."
    )

    parser.add_argument(
        '--save_only', action='store_true',
        help="If set, suppresses grid and node outputs."
    )

    return parser


def parse_output_arg(output_str):
    try:
        file, number_str = output_str.split(';')
        number = int(number_str)
        if number <= 0:
            raise ValueError("The number of files must be greater than 0.")
        return file, number
    except ValueError:
        raise ValueError(
            f"Invalid output format: '{output_str}'. "
            "Expected format is 'filename.npz;<number>'."
        )


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        file, number = parse_output_arg(args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    mode = 'ratio' if args.ratio is not None else 'number'
    value = args.ratio if args.ratio is not None else args.number

    if mode == 'ratio' and not (0 <= value <= 1):
        print("Error: Ratio must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    for i in range(number):
        grid, G = (
            generate_grid(args.size, 'ratio', value)
            if mode == 'ratio'
            else generate_grid(args.size, 'number', value, value)
        )

        save_graph(G, f"{file}_{i}.npz")

        if not args.save_only:
            print(f"Graph {i + 1}/{number} saved as '{file}_{i}.npz'")
            print("Generated grid:")
            print(grid)
            print("Graph nodes:", list(G.nodes))


if __name__ == "__main__":
    main()
