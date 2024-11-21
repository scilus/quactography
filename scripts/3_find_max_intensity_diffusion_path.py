import argparse
import sys

sys.path.append(r"C:\Users\harsh\quactography")

from quactography.graph.undirected_graph import Graph
from quactography.adj_matrix.io import load_graph
from quactography.hamiltonian.hamiltonian_qubit_edge import Hamiltonian_qubit_edge
from quactography.solver.qaoa_solver_qu_edge import multiprocess_qaoa_solver_edge

# from quactography.hamiltonian.hamiltonian_qubit_node import Hamiltonian_qubit_node
# from quactography.solver.qaoa_solver_qu_node import multiprocess_qaoa_solver_node


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
        "hamiltonian",
        help="Hamiltonian qubit representation to use for QAOA, either 'node' or 'edge' ",
        type=str,
    )
    p.add_argument(
        "--alphas", nargs="+", type=float, help="List of alphas", default=[1.1]
    )

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

    p.add_argument(
        "--optimizer",
        help="Optimizer to use for the QAOA algorithm",
        default="Powell",
        type=str,
    )

    p.add_argument(
        "--refinement_loops",
        help="Number of loops for the refinement optimization",
        default=3,
        type=int,
    )

    p.add_argument(
        "--epsilon",
        help="Epsilon value for the refinement optimization",
        default=1e-5,
        type=float,
    )

    return p


def main():
    """
    Uses QAOA with multiprocess as an option to find shortest path, with a given Graph, starting, ending node and Hamiltonian associated
    to the graph.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, _, _ = load_graph(args.in_graph + ".npz")
    graph = Graph(weighted_graph, args.starting_node, args.ending_node)

    # Construct Hamiltonian when qubits are set as edges, then optimize with QAOA/scipy:
    if args.hamiltonian == "edge":

        hamiltonians = [Hamiltonian_qubit_edge(graph, alpha) for alpha in args.alphas]

        print(hamiltonians[0].total_hamiltonian.simplify())

        print("\n Calculating qubits as edges......................")
        multiprocess_qaoa_solver_edge(
            hamiltonians,
            args.reps,
            args.number_processors,
            args.output_file,
            args.optimizer,
            args.refinement_loops,
            args.epsilon,
        )

    # # Construct Hamiltonian when qubits are set as nodes, then optimize with QAOA/scipy:
    # elif args.hamiltonian == "node":
    #     hamiltonians = [Hamiltonian_qubit_node(graph, alpha) for alpha in args.alphas]
    #     print(hamiltonians[0].total_hamiltonian.simplify())

    #     print("\n Calculating qubits as nodes......................")
    #     multiprocess_qaoa_solver_node(
    #         hamiltonians,
    #         args.reps,
    #         args.number_processors,
    #         args.output_file,
    #         # args.optimizer,
    #         # args.refinement_loops,
    #         # args.epsilon,
    #     )


if __name__ == "__main__":
    main()
