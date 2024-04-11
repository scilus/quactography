import argparse

from quactography.graph.graph_with_connexions import Graph
from quactography.adj_matrix.io import load_graph
from quactography.hamiltonian.hamiltonian_total import Hamiltonian
from quactography.solver.multiprocess_solver import multiprocess_qaoa_solver
from quactography.solver.qaoa_solver import _find_longest_path
from quactography.visu.dist_prob import _plot_distribution_of_probabilities
from quactography.visu.multiprocess_visu import multiprocess_visu


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter  # type: ignore
    )
    p.add_argument(
        "in_graph",
        help="Adjacency matrix which graph we want path that maximizes weights in graph",
        type=str,
    )
    p.add_argument("starting_node", help="Starting node of the graph", type=int)
    p.add_argument("ending_node", help="Ending node of the graph", type=int)
    p.add_argument("output_file", help="Output file name", type=str)

    p.add_argument(
        "-a", "--alphas", nargs="+", type=int, help="List of alphas", default=[1.1]
    )
    p.add_argument("-m", "--multiprocess", help="Use multiprocess", action="store_true")

    p.add_argument(
        "-v",
        "--visual_dist_output_file_total",
        help="Output file name for visualisation",
        type=str,
    )
    p.add_argument(
        "-vs",
        "--visual_dist_output_file_selected",
        help="Output file name for visualisation",
        type=str,
    )
    p.add_argument(
        "-d", "--dist_show", help="Show probabilities distribution", action="store_true"
    )
    p.add_argument(
        "-r",
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

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, _, _ = load_graph(args.in_graph)

    graph = Graph(weighted_graph, args.starting_node, args.ending_node)
    hamiltonians = [Hamiltonian(graph, alpha) for alpha in args.alphas]

    if args.multiprocess:
        multiprocess_qaoa_solver(
            hamiltonians, args.reps, args.number_processors, args.output_file
        )
        if args.dist_show:
            multiprocess_visu(
                [
                    args.output_file + "_alpha_" + str(hamiltonians[i].alpha) + ".npz"
                    for i in range(len(hamiltonians))
                ],
                args.number_processors,
                args.visual_dist_output_file_total,
                args.visual_dist_output_file_selected,
                hamiltonians,
            )
        # Ajouter visualisation de la distribution de probabilités pour multiprocess
        # Ajouter visualisation des chemins pour multiprocess

    else:  # Pour une seule valeur de alpha
        for i in range(len(hamiltonians)):
            _find_longest_path([hamiltonians[i], args.reps, args.output_file + str(i)])

            if args.dist_show:
                _plot_distribution_of_probabilities(
                    [
                        args.output_file
                        + "_alpha_"
                        + str(hamiltonians[i].alpha)
                        + ".npz",
                        args.visual_dist_output_file_total,
                        args.visual_dist_output_file_selected,
                        hamiltonians[i],
                    ]
                )
            # Ajouter visualisation des chemins pour une seule valeur de alpha


if __name__ == "__main__":
    main()
