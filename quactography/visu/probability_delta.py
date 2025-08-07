import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from pathlib import Path
from quactography.solver.io import load_optimization_results


def prob_delta_for_reps(
    in_folder,
    out_file,
    save_only
):
    """
    Visualize the distance of the optimal path from the most probable path
    for different repetitions in terms of probabilities. 
    i.e. How much more or less probable the optimal path is compared to the most probable path.

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
    
    delta_dict = defaultdict(list)
    
    #load optimization results from all files in the directory
    path = Path(in_folder)
    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        _, dist_prob, min_cost, h, _, rep, _ = load_optimization_results(in_file_path)
        min_cost = min_cost.item()
        dist_prob = dist_prob.item()
        h = h.item()
        rep = rep.item()

        # calculate the delta from the optimal path
        # exact_path is the theorical optimal path found with classical means
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_prob))))
        maxp = max(dist_prob, key=dist_prob.get)
        maxp = dist_prob[maxp]
        delta = maxp - dist_prob[exact_path]

        # store the delta for each repetition
        delta_dict[rep].append(delta)

    
    delta_dict = OrderedDict(sorted(delta_dict.items()))
    labels, data = [*zip(*delta_dict.items())]


    plt.boxplot(data, notch=False)
    plt.xticks(range(1, len(labels) + 1),labels)
    plt.xlabel("Repetitions")
    plt.ylabel("QAOA delta")
    plt.title("Delta vs repetitions")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_reps.png")
    print("Visualisation of the distance form optimal energy for different seeds"
            f"and repetitions on identical alphas saved in {out_file}_reps.png")

    plt.close()


def prob_delta_for_alpha(
    in_folder,
    out_file,
    save_only
):
    """
    Visualize the distance of the optimal path from the most probable path
    for different alphas in terms of probabilities. 
    i.e. How much more or less probable the optimal path is compared to the most probable path.
    
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
    
    delta_dict = defaultdict(list)
    
    #load optimization results from all files in the directory
    path = Path(in_folder)
    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        _, dist_prob, min_cost, h, _, _, _ = load_optimization_results(in_file_path)
        min_cost = min_cost.item()
        dist_prob = dist_prob.item()
        h = h.item()
        alpha = h.alpha_init

        # calculate the delta from the optimal path
        # exact_path is the theorical optimal path found with classical means
        exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_prob))))
        maxp = max(dist_prob, key=dist_prob.get)
        maxp = dist_prob[maxp]
        delta = (maxp - dist_prob[exact_path])
        
        # store the delta for each alpha
        delta_dict[alpha].append(delta)

    
    delta_dict = OrderedDict(sorted(delta_dict.items()))
    labels, data = [*zip(*delta_dict.items())]



    plt.boxplot(data, notch=False)
    plt.xticks(range(1, len(labels) + 1),labels)
    plt.xlabel("alphas")
    plt.ylabel("QAOA Delta")
    plt.title("Delta vs alphas")

    if not save_only:
        plt.show()

    plt.savefig(f"{out_file}_alphas.png")
    print("Visualisation of the distance from optimal energy for different seeds"
          f" and alphas on uniform repetition saved in {out_file}_alphas.png")

    plt.close()
