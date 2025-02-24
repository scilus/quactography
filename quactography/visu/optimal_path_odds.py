import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from pathlib import Path
from quactography.solver.io import load_optimization_results
from qiskit.visualization import plot_distribution


def visualize_optimal_prob_rep(
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

    probs = []
    path = Path(in_file)
    hprobs = []
    reps = []
    glob_path = path.glob('*')

    for in_file_path in glob_path:
        _, dist_binary_prob, _, h, bin_str, rep, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        opt_path = bin_str.item()
        h = h.item()
        probs.append(dist_binary_prob[opt_path])
        exact_path = h.exact_path[0].zfill(11)
        hprobs.append(dist_binary_prob[exact_path])
        reps.append(rep)

    plt.scatter(reps, probs)
    plt.scatter(reps, hprobs)
    plt.xlabel("Repitition")
    plt.ylabel("Quasi-probability")
    plt.title("Prob vs reps")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_prob_for_reps.png")
    print("Visualisation of the distance form optimal energy for different seeds "
            f"and repetitions on identical alphas saved in {out_file}_prob_reps.png")


def visualize_optimal_prob_alpha(
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
    path = Path(in_file)
    probs = []
    hprobs = []

    glob_path = path.glob('*')

    for in_file_path in glob_path:
        _, dist_binary_prob, _, h, bin_str, _, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        opt_path = bin_str.item()
        h = h.item()
        probs.append(dist_binary_prob[opt_path])
        exact_path = h.exact_path[0].zfill(11)
        hprobs.append(dist_binary_prob[exact_path])
        alphas.append(h.alpha)

    plt.scatter(alphas, probs)
    plt.scatter(alphas, hprobs)
    plt.xlabel("alphas")
    plt.ylabel("Quasi-probability")
    plt.title("Prob vs alphas")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_prob_for_alphas.png")
    print("Visualisation of the distance from optimal energy for different seeds"
          f" and alphas on uniform repetition saved in {out_file}_prob_for_alphas.png")

    plt.close()


def visualize_optimal_paths_prob(
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

    paths = []
    path = Path(in_file)
    sumDir = 0
    glob_path = path.glob('*')

    for in_file_path in glob_path:
        path = []
        mercy = {}
        _, dist_binary_prob, _, h, bin_str, _, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        opt_path = bin_str.item()
        h = h.item()
        exact_path = h.exact_path[0].zfill(11)
        path.append({opt_path: dist_binary_prob[opt_path]})
        path.append({exact_path: dist_binary_prob[exact_path]})
        for key in path:
            mercy.update(key)
        paths.append({key: mercy[key] for key in mercy})
        sumDir += 1

    legend = []
    colors = []
    last_key = list(paths[0])[-1]
    print(list(paths[0])[-1])
    color = iter(cm.rainbow(numpy.linspace(0, 1, sumDir+1)))

    for j in range(sumDir):
        legend.append("File_" + str(j+1))
        colors.append(next(color))

    plot_distribution(
        paths,
        figsize=(14, 10),
        title="Distribution of probabilities",
        sort="hamming",
        color=colors,
        target_string=last_key
    )

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_prob_for_reps.png")
    print("Visualisation of the distance form optimal energy for different seeds "
            f"and repetitions on identical alphas saved in {out_file}_prob_reps.png")
