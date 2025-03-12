import matplotlib.pyplot as plt

from pathlib import Path
from quactography.solver.io import load_optimization_results


def visualize_optimal_paths_edge_rep(
    in_folder,
    out_file,
    save_only
):
    """
    Visualize the optimal path on a graph.

    Parameters
    ----------

    in_folder: str
        The folder containing the optimization results in .npz format
    out_file: str
        The output file name for the visualisation in .png format.
    save_only: bool
        If True, the figure is saved without displaying it
    Returns
    -------
    None """
    reps = []
    alphas = []
    square_loss = []
    param_count = 0
    path = Path(in_folder)

    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        _, _, min_cost, h, _, rep, _ = load_optimization_results(in_file_path)
        min_cost = min_cost.item()
        h = h.item()
        alpha = h.alpha * h.graph.number_of_edges/h.graph.all_weights_sum

        if alpha not in alphas:
            param_count += 1

        alphas.append(alpha)
        reps.append(rep)
        square_loss.append((min_cost + h.exact_cost)**2)



    scatter = plt.scatter(reps, square_loss, c=alphas)
    plt.legend(*scatter.legend_elements(num=param_count-1), 
               loc="upper right", title="Alphas")
    plt.xlabel("Repetitions")
    plt.ylabel("Square loss")
    plt.title("Square loss vs repetitions")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_alpha_{alpha:.2f}.png")
    print("Visualisation of the distance form optimal energy for different seeds"
            f"and repetitions on identical alphas saved in {out_file}_alpha_{alpha:.2f}.png")

    plt.close()


def visualize_optimal_paths_edge_alpha(
    in_folder,
    out_file,
    save_only
):
    """
    Visualize the optimal path on a graph.

    Parameters
    ----------

    in_folder: str
        The folder containing the optimization results in .npz format
    out_file: str
        The output file name for the visualisation in .png format.
    save_only: bool
        If True, the figure is saved without displaying it
    Returns
    -------
    None """
    reps = []
    alphas = []
    square_loss = []
    param_count = 0
    path = Path(in_folder)

    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        _, _, min_cost, h, _, rep, _ = load_optimization_results(in_file_path)
        min_cost = min_cost.item()
        h = h.item()
        alpha = h.alpha * h.graph.number_of_edges/h.graph.all_weights_sum
        alphas.append(alpha)

        if rep not in reps:
            param_count += 1

        reps.append(rep)
        square_loss.append((min_cost - h.exact_cost)**2)
        

    scatter = plt.scatter(alphas, square_loss, c=reps)
    plt.legend(*scatter.legend_elements(num=param_count-1))
    plt.xlabel("alphas")
    plt.ylabel("Square loss")
    plt.title("Square loss vs alphas")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_rep_{rep}.png")
    print("Visualisation of the distance from optimal energy for different seeds"
          f" and alphas on uniform repetition saved in {out_file}_rep_{rep}.png")

    plt.close()
