import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd 
import seaborn as sns 

from pathlib import Path
from quactography.solver.io import load_optimization_results


def visualize_optimal_prob_rep(
    in_folder,
    out_file,
    save_only
):
    """
    Visualize a scatter plot of the optimal path and the exact path for different repetitions.

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

    probs = []
    path = Path(in_folder)
    hprobs = []
    reps = []
    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        _, dist_binary_prob, _, h, bin_str, rep, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        opt_path = bin_str.item()[::-1]
        h = h.item()
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_binary_prob))))
 
        probs.append(dist_binary_prob[opt_path])
        hprobs.append(dist_binary_prob[exact_path])
        reps.append(rep)

    plt.scatter(reps, probs).set_label('Optimal path')
    plt.scatter(reps, hprobs).set_label('Exact path')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Repitition")
    plt.ylabel("Quasi-probability")
    plt.title("Prob vs reps")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_prob_for_reps.png")
    print("Visualisation of the distance form optimal energy for different seeds "
            f"and repetitions on identical alphas saved in {out_file}_prob_reps.png")


def visualize_optimal_prob_alpha(
    in_folder,
    out_file,
    save_only
):
    """
    Visualize a scatter plot of the optimal path and the exact path for different alphas.

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
    alphas = []
    path = Path(in_folder)
    probs = []
    hprobs = []

    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        _, dist_binary_prob, _, h, bin_str, _, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        opt_path = bin_str.item()[::-1]
        h = h.item()
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_binary_prob))))

        probs.append(dist_binary_prob[opt_path])
        hprobs.append(dist_binary_prob[exact_path])
        alphas.append(h.alpha)

    plt.scatter(alphas, probs).set_label('Optimal path')
    plt.scatter(alphas, hprobs).set_label('Exact path')
    plt.legend()
    plt.grid(True)
    plt.xlabel("alphas")
    plt.ylabel("Quasi-probability")
    plt.title("Prob vs alphas")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_prob_for_alphas.png")
    print("Visualisation of the distance from optimal energy for different seeds"
          f" and alphas on uniform repetition saved in {out_file}_prob_for_alphas.png")

    plt.close()

    
def visu_heatmap(
   in_folder,
    out_file,
    save_only
):
    """
    Visualize a heatmap of the optimal path  according to alphas and repetitions given.

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
    path = Path(in_folder)
    alphas = []
    heat = []
    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        path = []
        _, dist_binary_prob, _, h, _, rep, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        h = h.item()
        alpha = h.alpha * h.graph.number_of_edges/h.graph.all_weights_sum
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_binary_prob))))

        if dist_binary_prob[exact_path] not in heat:
            reps.append(rep.item())
            alphas.append(alpha)
            heat.append(dist_binary_prob[exact_path])

    df = pd.DataFrame.from_dict(np.array([reps,alphas,heat]).T)
    df.columns = ['Repitition', 'Alphas','Probability of optimal path']
    df['Probability of optimal path'] =pd.to_numeric(df["Probability of optimal path"])

    pivotted = df.pivot(index='Alphas',columns='Repitition',values='Probability of optimal path')

    heatmap = sns.heatmap(pivotted,cmap='RdBu')
    fig = heatmap.get_figure()
    fig.savefig(out_file+"_heatmap")
    print("Visualisation of the heatmap of the optimal path according to alpha and repetition "
            f"and repetitions on identical alphas saved in {out_file}_heatmap_.png")
