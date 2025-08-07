#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from quactography.graph.undirected_graph import Graph
from quactography.adj_matrix.io import load_graph
from quactography.hamiltonian.hamiltonian_qubit_edge import Hamiltonian_qubit_edge
from quactography.solver.qaoa_solver_qu_edge import multiprocess_qaoa_solver_edge, multiprocess_qaoa_solver_edge_rap


"""
Tool to run QAOA, optimize parameters, plot cost landscape with optimal
parameters found if only one reps, and returns the optimization results.
"""

def rap_funct(weighted_graph, starting_node,ending_node, alphas,
                reps, number_processors=2, optimizer="Differential"):
    """
    Process he Graph in order to create the Hamiltonian matrix before optimization
    with QAOA algorithm. The Hamiltonian is constructed with qubits as edges.

    Parameters
    ----------  
    graph : str
        Path to the input graph file (npz file).
    starting_node : int
        Starting node of the graph.
    ending_node : int
        Ending node of the graph.
    alphas : list of float, optional
        List of alpha values for the Hamiltonian. Default is [1.2].
    reps : int, optional
        Number of repetitions for the QAOA algorithm, determines the number
        of sets of gamma and beta angles. Default is 1.
    number_processors : int, optional
        Number of CPU to use for multiprocessing. Default is 2.
    optimizer : str, optional
        Optimizer to use for the QAOA algorithm. Default is "Differential".
    Returns
    -------
    line : list
        List of coordinates for the streamline.
    """
    graph = Graph(weighted_graph, starting_node, ending_node)

    # Construct Hamiltonian when qubits are set as edges,
    # then optimize with QAOA/scipy:

    hamiltonians = [Hamiltonian_qubit_edge(graph, alpha) for alpha in alphas]

    # print(hamiltonians[0].total_hamiltonian.simplify())

    print("\n Calculating qubits as edges......................")
    # Run the multiprocess QAOA solver for edge qubits
    # and return the optimal path as a list of coordinates.
    line = multiprocess_qaoa_solver_edge_rap(
        hamiltonians,
        reps,
        number_processors,
        graph.number_of_edges,
        optimizer
        )
    return line

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
    p.add_argument("output_directory",
                    help="Directory where the files will be outputed", type=str,
                    default="data/output_graphs/"
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        help="List of alphas",
        default=[1.2]
    )
    p.add_argument(
        "--reps",
        nargs="+",
        type=int,
        help="List of repetitions to run for the QAOA algorithm",
        default=[1],
    )
    p.add_argument(
        "-npr",
        "--number_processors",
        help="Number of cpu to use for multiprocessing",
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
    for i in range(len(args.reps)):
        multiprocess_qaoa_solver_edge(
            hamiltonians,
            args.reps[i],
            args.number_processors,
            args.output_file,
            args.output_directory,
            args.optimizer,
            args.plt_cost_landscape,
            args.save_only,
            )


if __name__ == "__main__":
    main()
