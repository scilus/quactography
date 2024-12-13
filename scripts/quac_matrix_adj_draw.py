import argparse
import numpy as np

from quactography.adj_matrix.io import load_graph
import matplotlib.pyplot as plt

"""Tool to visualize the graph constructed with diffusion data (white matter mask and fodf peaks)"""
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_graph", help="Graph file as a .npz archive.")
    return p


def main():
    """
    Visualize graph using matplotlib.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, node_indices, vol_dim = load_graph(args.in_graph)
    x, y = np.unravel_index(node_indices, vol_dim)
    weighted_graph = np.triu(weighted_graph)

    # draw the graph
    for it, node_row in enumerate(weighted_graph):
        nb_adj = np.count_nonzero(node_row)
        if nb_adj > 0:
            w_all = node_row[node_row > 0]
            start_x, start_y = x[it], y[it]
            end_x = x[node_row > 0]
            end_y = y[node_row > 0]
            for vert_id in range(nb_adj):
                w = w_all[vert_id]
                alpha = np.clip(w * 0.9 + 0.1, 0.0, 1.0)
                plt.plot(
                    [start_x, end_x[vert_id]],
                    [start_y, end_y[vert_id]],
                    color="black",
                    alpha=alpha,
                )

    plt.scatter(x, y)
    plt.savefig("graph_adj_mat.png")  # Save the plot as a PNG file
    plt.show()


if __name__ == "__main__":
    main()
