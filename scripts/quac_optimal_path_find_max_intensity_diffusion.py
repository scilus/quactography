#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from quactography.graph.undirected_graph import Graph
from quactography.adj_matrix.io import load_graph
from quactography.hamiltonian.hamiltonian_qubit_edge import Hamiltonian_qubit_edge
from quactography.solver.qaoa_solver_qu_edge import multiprocess_qaoa_solver_edge


"""
Tool to run QAOA, optimize parameters, plot cost landscape with optimal
parameters found if only one reps, and returns the optimization results.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "in_graph",
        help="Adjacency matrix which graph we want path that maximizes weights in graph, (npz file)",
        type=str,
    )
    p.add_argument("starting_node",
                   help="Starting node of the graph", type=int)
    p.add_argument("ending_node",
                   help="Ending node of the graph", type=int)
    p.add_argument("output_file",
                   help="Output file name (npz file)", type=str)
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        help="List of alphas",
        default=[1.2]
    )
    p.add_argument(
        "--reps",
        help="Number of repetitions for the QAOA algorithm",
        type=int,
        default=1,
    )
    p.add_argument(
        "-npr",
        "--number_processors",
        help="number of cpu to use for multiprocessing",
        default=1,
        type=int,
    )
    p.add_argument(
        "--optimizer",
        help="Optimizer to use for the QAOA algorithm",
        default="Differential",
        type=str,
    )
    p.add_argument(
        "--plt_cost_landscape",
        help="True or False, Plot 3D and 2D of the cost landscape"
        "(for gamma and beta compact set over all possible angles-0.1 incrementation)",
        action="store_false",
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

    weighted_graph, _, _ = load_graph(args.in_graph)
    graph = Graph(weighted_graph, args.starting_node, args.ending_node)

    # Construct Hamiltonian when qubits are set as edges,
    # then optimize with QAOA/scipy:

    hamiltonians = [Hamiltonian_qubit_edge(graph, alpha) for alpha in args.alphas]

    # print(hamiltonians[0].total_hamiltonian.simplify())

    print("\n Calculating qubits as edges......................")
    multiprocess_qaoa_solver_edge(
        hamiltonians,
        args.reps,
        args.number_processors,
        args.output_file,
        args.optimizer,
        args.plt_cost_landscape,
        args.save_only,
    )


if __name__ == "__main__":
    main()
