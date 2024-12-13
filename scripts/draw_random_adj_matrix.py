import rustworkx as rx
import matplotlib.pyplot as plt
from rustworkx.visualization import mpl_draw as draw
import matplotlib.pyplot as plt
import argparse

from quactography.adj_matrix.io import load_graph


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_graph", help="Graph file as a .npz archive.")
    p.add_argument(
        "initial_graph_filename", help="Initial graph visual filename.", type=str
    )
    return p


def main():
    """
    Visualize graph using rustworkx and matplotlib.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, node_indices, _ = load_graph(args.in_graph + ".npz")
    graph = rx.PyGraph(multigraph=False)
    num_nodes = len(node_indices)
    nodes_list = graph.add_nodes_from((range(num_nodes)))
    # Add edges :
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if weighted_graph[i, j] != 0:
                edges.append((i, j, ((weighted_graph[i, j])).round(3)))

    graph.add_edges_from(edges)
    draw(graph, with_labels=True, edge_labels=str, pos=rx.graph_spring_layout(graph))  # type: ignore

    # Save figure in output
    plt.savefig(args.initial_graph_filename)
    plt.close()


if __name__ == "__main__":
    main()
