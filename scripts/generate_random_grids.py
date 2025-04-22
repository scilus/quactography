#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate random grids/graphs with obstacles.
This script can optionally suppress the output
display of the grid and graph nodes.
The generated graphs are saved as .npz files.
"""

import argparse
from my_research.utils.grid_dijkstra import generer_grille, save_graph


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--size', type=int, default=10,
        help="Size of the grid (grid will be of shape size x size)."
    )

    parser.add_argument(
        '--obstacles', type=str, default='ratio:0.2',
        help="Obstacle settings: 'ratio:<value>' or 'number:<value>'"
    )

    parser.add_argument(
        '--output', required=True,
        help="Output format: 'filename.npz;<number>'. "
             "This will generate <number> files like 'filename_0.npz', etc."
    )

    parser.add_argument(
        '--save_only', action='store_true',
        help="If set, suppress grid and node outputs."
    )

    return parser


def parse_obstacle_mode(obstacle_str):
    mode, value_str = obstacle_str.split(':')
    value = float(value_str) if mode == 'ratio' else int(value_str)

    if mode == 'ratio' and not (0 <= value <= 1):
        raise ValueError(
            f"The obstacle ratio must be between 0 and 1 (received: {value})")

    return mode, value


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    mode, value = parse_obstacle_mode(args.obstacles)

    file, number = args.output.split(';')
    number = int(number)

    grid, G = (
        generer_grille(args.size, mode, value)
        if mode == "ratio"
        else generer_grille(args.size, mode, value, value)
    )

    if not args.save_only:
        print(f"{number} graphs saved as '{file}_X.npz'.")
        print("Grille générée :")
        print(grid)
        print("Graph nodes:", list(G.nodes))

    for i in range(number):
        save_graph(G, f"{file}_{i}.npz")


if __name__ == "__main__":
    main()
