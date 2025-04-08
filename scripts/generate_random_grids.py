#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate random grids/graphs with obstacles.
"""

import argparse
from my_research.utils.grid_dijkstra import generer_grille, save_graph


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        '--size', type=int, default=10,
        help="Size of the grid (grid will be of shape size x size)."
    )
    p.add_argument(
        '--obstacles', type=str, default='ratio:0.2',
        help="Obstacle settings: 'ratio:<value>' or 'number:<value>'"
    )
    p.add_argument(
        '--output', type=str, required=True,
        help="Output format: 'filename.json;<number>'. "
             "This will generate <number> files like 'filename_0.json', etc."
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    mode, value = args.obstacles.split(':')
    value = float(value) if mode == 'ratio' else int(value)

    if mode == 'ratio' and not (0 <= value <= 1):
        raise ValueError(
            "The obstacle ratio must be between 0 and 1"
            f"(received: {args.obstacles})"
        )

    file, number = args.output.split(';')
    number = int(number)

    # Générer une seule grille pour tous les graphes
    grid, G = (
        generer_grille(args.size, mode, value)
        if mode == "ratio"
        else generer_grille(args.size, mode, value, value)
    )

    for i in range(number):
        save_graph(G, f"{file}_{i}.json")

    print(f"✅ {number} graphs saved as '{file}_X.json'.")
    print("Grille générée :")
    print(grid)
    print("Graph nodes:", list(G.nodes))


if __name__ == "__main__":
    main()
