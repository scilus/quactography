import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from quactography.solver.io import load_optimization_results
from quactography.adj_matrix.io import load_graph


def visualize_optimal_paths_edge_rep(
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
    reps = []
    square_loss = []
    path = Path(in_file)

    glob_path = path.glob('*')

    for in_file_path in glob_path:
        _, _, min_cost, h, bin_str, rep, opt_params = load_optimization_results(in_file_path)
        min_cost = min_cost.item()
        h = h.item()
        alpha = h.alpha
        print(h.exact_cost)
        print(min_cost)
        reps.append(rep)
        square_loss.append((min_cost + h.exact_cost)**2)
        
    plt.scatter(reps,square_loss)
    plt.xlabel("Repetitions")
    plt.ylabel("Square loss")
    plt.title("Square loss vs repetitions")
        

    # plt.show()
    # plt.tight_layout()
    
    # plt.legend(
    #     [
    #         f"alpha_factor = {(alpha):.2f},\n Cost: {min_cost:.2f}\n "
    #           f"\n reps : {reps},\n Actual path : {bin_str} "
            
    #     ],
    #     loc="upper right",
    # )
    # if not save_only:
    #     plt.show()
    if not save_only:
        plt.show()
   
    plt.savefig(f"{out_file}_alpha_{alpha:.2f}.png")
    print("Visualisation of the distance form optimal energy for different seeds"
           f"and repetitions on identical alphas saved in {out_file}_alpha_{alpha:.2f}.png")
    
    plt.close()

def visualize_optimal_paths_edge_alpha(
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
    alphas = []
    square_loss = []
    path = Path(in_file)

    glob_path = path.glob('*')

    for in_file_path in glob_path:
        _, _, min_cost, h, _, rep, _ = load_optimization_results(in_file_path)
        min_cost = min_cost.item()
        h = h.item()
        alpha = h.alpha

        alphas.append(alpha)
        square_loss.append((min_cost - h.exact_cost)**2)
        
    plt.scatter(alphas,square_loss)
    plt.xlabel("alphas")
    plt.ylabel("Square loss")
    plt.title("Square loss vs alphas")
        

    # plt.show()
    # plt.tight_layout()
    
    # plt.legend(
    #     [
    #         f"alpha_factor = {(alpha):.2f},\n Cost: {min_cost:.2f}\n "
    #           f"\n reps : {reps},\n Actual path : {bin_str} "
            
    #     ],
    #     loc="upper right",
    # )
    # if not save_only:
    #     plt.show()
    if not save_only:
        plt.show()
   
    plt.savefig(f"{out_file}_rep_{rep}.png")
    print("Visualisation of the distance from optimal energy for different seeds"
          f" and alphas on uniform repetition saved in {out_file}_rep_{rep}.png")
    
    plt.close()