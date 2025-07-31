#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find the shortest path between 2 points
in a graph using Dijkstra or A* algorithm.
Supports graphs loaded from JSON or NPZ files,
and optionally allows diagonal movement.
"""

import argparse
import os
import sys
from quactography.classical.utils.random_Dijkstra import dijkstra_stepwise
from quactography.classical.utils.random_Astar import astar_stepwise, heuristic
from quactography.classical.io import load_the_graph


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--in_graph", type=str, required=True,
        help="Path to the input graph file (.json or .npz)"
    )
    parser.add_argument(
        "--shortestpath", choices=['Dijkstra', 'A*'], default='Dijkstra',
        help="Shortest path algorithm to use: 'Dijkstra' or 'A*'"
    )
    parser.add_argument(
        "--start", type=str, required=True,
        help="Start node, e.g. '3,4'"
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Target node, e.g. '7,8'"
    )
    parser.add_argument(
        "--diagonal_mode", choices=['diagonal', 'nondiagonal'],
        default='nondiagonal',
        help="Allow diagonal movement or not"
    )
    return parser


def parse_node(node_str):
    try:
        parts = node_str.strip().split(',')
        return tuple(int(p) for p in parts if p.strip() != '')
    except ValueError as e:
        raise ValueError(
            f"Invalid node format: '{node_str}' (expected format: x,y)"
        ) from e


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.in_graph):
        print(f"Error: File '{args.in_graph}' not found.")
        sys.exit(1)

    try:
        start = parse_node(args.start)
        target = parse_node(args.target)
    except ValueError as e:
        print(f"Error parsing node: {e}")
        sys.exit(1)

    G = load_the_graph(args.in_graph)

    if start not in G.nodes():
        print(f"Start node {start} not in graph.")
        print(f"Available nodes: {list(G.nodes())[:5]}...")
        sys.exit(1)

    if target not in G.nodes():
        print(f"Target node {target} not in graph.")
        print(f"Available nodes: {list(G.nodes())[:5]}...")
        sys.exit(1)

    print(
        f"Finding shortest path from {start} to {target} using {args.shortestpath}..."
        )

    if args.shortestpath == "Dijkstra":
        evaluated_nodes, path_history, path_cost = dijkstra_stepwise(
            G, start, target, args.diagonal_mode
        )
    else:
        evaluated_nodes, path_history, path_cost = astar_stepwise(
            G, start, target, args.diagonal_mode
        )

    if path_history is None:
        print("No path found.")
        sys.exit(0)

    shortest_path = [tuple(int(x) for x in n) for n in path_history[-1]]

    print("\nShortest path:")
    print(" → ".join(map(str, shortest_path)))
    print(f"Path cost: {path_cost:.2f}")
    print(f"Nodes evaluated: {len(evaluated_nodes)}")


if __name__ == "__main__":
    main()
