import argparse
from quactography.graph.graph_with_connexions import Graph
from quactography.adj_matrix.io import load_graph
from quactography.hamiltonian.hamiltonian_total import Hamiltonian


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

    p.add_argument(
        "-a", "--alphas", nargs="+", type=int, help="List of alphas", default=[1.1]
    )

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, _, _ = load_graph(args.in_graph)

    graph = Graph(weighted_graph, args.starting_node, args.ending_node)
    hamiltonians = [Hamiltonian(graph, alpha) for alpha in args.alphas]
    # for hamiltonian in hamiltonians:
    #     print(hamiltonian.total_hamiltonian)

    # reste qaoa solver et visualisation à faire


if __name__ == "__main__":
    main()
