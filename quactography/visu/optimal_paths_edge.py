import networkx as nx
import matplotlib.pyplot as plt

from quactography.solver.io import load_optimization_results
from quactography.adj_matrix.io import load_graph


def visualize_optimal_paths_edge(
    graph_file,
    in_file,
    out_file,
    save_only
):
    """
    Visualize the optimal path on a graph.

    Parameters
    ----------
    graph_file: str
        The input file containing the graph in .npz format.
    in_file: str
        The input file containing the optimization results in .npz format
    out_file: str
        The output file name for the visualisation in .png format.
    save_only: bool
        If True, the figure is saved without displaying it
    Returns
    -------
    None """
    _, _, min_cost, h, bin_str, reps, opt_params = load_optimization_results(in_file)
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
        pos=nx.shell_layout(G),
        edge_labels=edge_label,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
    )
    nx.draw_shell(
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
            f"alpha_factor = {(alpha):.2f},\n Cost: {min_cost:.2f}\n Starting node : {starting_node},
              \n Ending node : {ending_node},\n reps : {reps},\n Actual path : {bin_str} "
            
        ],
        loc="upper right",
    )
    if not save_only:
        plt.show()
   
    plt.savefig(f"{out_file}_alpha_{alpha:.2f}.png")
    print(f"Visualisation of the optimal path saved in {out_file}_alpha_{alpha:.2f}.png")
    
    plt.close()