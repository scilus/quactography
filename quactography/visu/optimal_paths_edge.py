import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx
import matplotlib
from rustworkx.visualization import mpl_draw
import argparse

from quactography.solver.io import load_optimization_results
from quactography.adj_matrix.io import load_graph


def visualize_optimal_paths_edge(
    graph_file,
    in_file,
    out_file,
):
    """_summary_ :Visualize the path taken in the graph and save the figure in the output folder

    Args:
        starting_nodes (list int): list of starting points
        ending_nodes (list int): list of ending_nodes points
        mat_adj (np array):  adjacency matrix
        list(map(int, bin_str)) (liste int): list of 0 and 1 representing the path taken
    """
    _, _, min_cost, h, bin_str, reps = load_optimization_results(in_file)
    min_cost = min_cost.item()
    h = h.item()
    bin_str = bin_str.item()
    reps = reps.item()
    mat_adj, _, _ = load_graph(graph_file)

    graph = h.graph
    starting_node = graph.starting_node
    ending_node = graph.ending_node
    starting_nodes = graph.starting_nodes
    ending_nodes = graph.ending_nodes
    all_weights_sum = graph.all_weights_sum
    alpha = h.alpha

    bin_str = list(map(int, bin_str))

    # Create a graph where the edges taken are in green and the edges not taken are in black
    G = nx.Graph()
    edges_taken = []
    edges_not_taken = []

    for i, value in enumerate(bin_str):
        if value == 1:
            edge_taken = (
                starting_nodes[i],
                ending_nodes[i],
                {
                    "weight": mat_adj[starting_nodes[i], ending_nodes[i]],
                    "color": "green",
                },
            )
            edges_taken.append(edge_taken)
        elif value == 0:
            edge_not_taken = (
                starting_nodes[i],
                ending_nodes[i],
                {
                    "weight": mat_adj[starting_nodes[i], ending_nodes[i]],
                    "color": "black",
                },
            )
            edges_not_taken.append(edge_not_taken)
    G.add_edges_from(edges_taken)
    G.add_edges_from(edges_not_taken)

    # Draw the graph
    edge_label = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G,
        pos=nx.planar_layout(G),
        edge_labels=edge_label,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
    )
    nx.draw_planar(
        G,
        with_labels=True,
        node_size=200,
        node_color="skyblue",
        font_size=10,
        edge_color=[G[u][v]["color"] for u, v in G.edges()],
        font_color="black",
        font_weight="bold",
    )
    # Add all elements in bin_str to a string
    bin_str = "".join(map(str, bin_str))
    # plt.show()
    plt.axis("off")
    # plt.tight_layout()
    plt.legend(
        [
            f"alpha_factor = {(alpha/all_weights_sum):.2f},\n Cost: {min_cost:.2f}\n Starting node : {starting_node}, \n Ending node : {ending_node},\n reps : {reps}, \n Good path : {h.exact_path}, \n Actual path : {bin_str}"
        ],
        loc="upper right",
    )
    # plt.show()
    plt.savefig(f"{out_file}_alpha_{alpha:.2f}.png")
    plt.close()


def minimize_crossings():
    """Create visulization of a graph with the minimum number of crossings with matplotlib"""
    pass
