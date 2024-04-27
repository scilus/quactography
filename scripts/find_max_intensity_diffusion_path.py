import argparse

from quactography.graph.graph_with_connexions import Graph
from quactography.adj_matrix.io import load_graph
from quactography.hamiltonian.hamiltonian_total import Hamiltonian
from quactography.solver.multiprocess_solver import multiprocess_qaoa_solver
from solver.qaoa import _find_longest_path


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "in_graph",
        help="Adjacency matrix which graph we want path that maximizes weights in graph, npz file",
        type=str,
    )
    p.add_argument("starting_node", help="Starting node of the graph", type=int)
    p.add_argument("ending_node", help="Ending node of the graph", type=int)
    p.add_argument("output_file", help="Output file name", type=str)

    p.add_argument(
        "--alphas", nargs="+", type=int, help="List of alphas", default=[1.1]
    )
    p.add_argument("--multiprocess", help="Use multiprocess", action="store_true")

    p.add_argument(
        "--reps",
        help="Number of repetitions for the QAOA algorithm",
        type=int,
        default=1,
    )
    # Voir avec scilpy :
    p.add_argument(
        "-npr",
        "--number_processors",
        help="number of cpu to use for multiprocessing",
        default=1,
        type=int,
    )

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, _, _ = load_graph(args.in_graph + ".npz")

    graph = Graph(weighted_graph, args.starting_node, args.ending_node)
    hamiltonians = [Hamiltonian(graph, alpha) for alpha in args.alphas]

    if args.multiprocess:
        multiprocess_qaoa_solver(
            hamiltonians, args.reps, args.number_processors, args.output_file
        )

    else:  # Pour une seule valeur de alpha
        for i in range(len(hamiltonians)):
            _find_longest_path(
                [hamiltonians[i], args.reps, args.output_file + "_" + str(i)]
            )


if __name__ == "__main__":
    main()
