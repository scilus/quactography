import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib  
matplotlib.use("Agg")
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
    None 
    """

    probs = []
    hprobs = []
    reps_minus = []
    reps_plus = []
    
    #load optimization results from all files in the directory
    path = Path(in_folder)
    glob_path = path.glob('*.npz')
    for in_file_path in glob_path:
        _, dist_binary_prob, _, h, bin_str, rep, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        opt_path = bin_str.item()[::-1]
        h = h.item()
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_binary_prob))))
 
        # set apart the optimal path and the exact path
        # opt_path is the path found with QAOA
        probs.append(dist_binary_prob[opt_path])
        hprobs.append(dist_binary_prob[exact_path])
        reps_minus.append(rep - 0.2)
        reps_plus.append(rep + 0.2)



    qaoa = plt.scatter(reps_minus, probs,marker=",")
    qaoa.set_label('QAOA solution')
    plt.scatter(reps_plus, hprobs).set_label('Strue solution')
    for i in range (len(reps_minus)):
        plt.plot([reps_minus[i],reps_plus[i]],[probs[i],hprobs[i]], 'k--')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Repetition")
    plt.xticks(np.arange(len(reps_minus)),np.arange(len(reps_minus)))

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
    None 
    """
    alphas_minus = []
    alphas_plus = []
    probs = []
    hprobs = []
    
    #load optimization results from all files in the directory
    path = Path(in_folder)
    glob_path = path.glob('*.npz')
    for in_file_path in glob_path:
        _, dist_binary_prob, _, h, bin_str, _, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        opt_path = bin_str.item()[::-1]
        h = h.item()
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_binary_prob))))

        # set apart the optimal path and the exact path
        # opt_path is the path found with QAOA
        probs.append(dist_binary_prob[opt_path])
        hprobs.append(dist_binary_prob[exact_path])
        alphas_minus.append(h.alpha_init - 0.15)
        alphas_plus.append(h.alpha_init + 0.15)

    plt.scatter(alphas_minus, probs,marker=",").set_label('QAOA solution')
    plt.scatter(alphas_plus, hprobs).set_label('True solution')
    for i in range (len(alphas_minus)):
        plt.plot([alphas_minus[i],alphas_plus[i]],[probs[i],hprobs[i]],'k--')
    plt.legend()
    plt.grid(True)
    plt.xlabel("alphas")
    plt.xticks(np.arange(len(alphas_minus)),np.arange(len(alphas_minus)))
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
    None 
    """

    reps = []
    alphas = []
    heat = []
    pos = []
    
    #load optimization results from all files in the directory
    path = Path(in_folder)
    glob_path = path.glob('*.npz')
    for in_file_path in glob_path:
        path = []
        _, dist_binary_prob, _, h, _, rep, _ = load_optimization_results(in_file_path)
        dist_binary_prob = dist_binary_prob.item()
        h = h.item()
        alpha = h.alpha_norm * h.graph.number_of_edges/h.graph.all_weights_sum
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_binary_prob))))

        # Add the exact path to the heatmap if it is not already present
        if dist_binary_prob[exact_path] not in heat:
            reps.append(rep.item())
            alphas.append(alpha)
            heat.append(dist_binary_prob[exact_path])
            sorted_dict = sorted(
                dist_binary_prob.items(), key=lambda items: items[1], 
            )
            ind = sorted_dict.index((exact_path, dist_binary_prob[exact_path]))
            pos.append(len(sorted_dict)-ind)

    df = pd.DataFrame.from_dict(np.array([reps,alphas,heat]).T)
    df.columns = ['Repetition', 'Alphas','Probability of optimal path']
    df['Probability of optimal path'] =pd.to_numeric(df["Probability of optimal path"])

    pivotted = df.pivot(index='Alphas',columns='Repetition',values='Probability of optimal path')

    heatmap = sns.heatmap(pivotted,cmap='jet',annot=True)
    heatmap.set_title('Optimal path quasi-probability')
    fig = heatmap.get_figure()
    fig.savefig(out_file + "_heatmap")
    print("Visualisation of the heatmap of the optimal path according to alpha and repetition "
            f"and repetitions on identical alphas saved in {out_file}_heatmap_.png")
    fig.clf()
    
    df_pos = pd.DataFrame.from_dict(np.array([reps,alphas,pos]).T)
    df_pos.columns = ['Repetition', 'Alphas','Probability of optimal path']
    df_pos['Probability of optimal path'] =pd.to_numeric(df_pos["Probability of optimal path"])

    pivotted_pos = df_pos.pivot(index='Alphas',columns='Repetition',values='Probability of optimal path')

    heatmap_pos = sns.heatmap(pivotted_pos, cmap="jet_r",annot=True)
    heatmap_pos.set_title('Optimal path position in the distribution')
    fig_pos = heatmap_pos.get_figure()
    fig_pos.savefig(out_file + "_pos")
    print("Visualisation of the heatmap of the optimal path according to alpha and repetition "
            f"and repetitions on identical alphas saved in {out_file}_heatmap_.png")
    fig.clf()
