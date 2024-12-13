from qiskit.visualization import plot_distribution
import matplotlib.pyplot as plt

from quactography.solver.io import load_optimization_results


def plot_distribution_of_probabilities_edge(
    in_file, visu_out_file_total, visu_out_file_selected, save_only
):

    _, dist_binary_prob, min_cost, h, _, _, opt_params = load_optimization_results(
        in_file
    )
    # convert dist_binary_prob a dictionary
    dist_binary_prob = dist_binary_prob.item()
    min_cost = min_cost.item()
    h = h.item()
    # Plot distribution of probabilities:
    plot_distribution(
        dist_binary_prob,
        figsize=(10, 8),
        title="Distribution of probabilities",
        color="pink",
    )
    if not save_only:
        plt.show()
    # Save plot of distribution:
    plt.savefig(visu_out_file_total)
    print("Distribution of probabilities saved in ", visu_out_file_total)

    # print(max(dist_binary_prob, key=dist_binary_prob.get))
    # bin_str = list(map(int, max(dist_binary_prob, key=dist_binary_prob.get)))
    # bin_str_reversed = bin_str[::-1]
    # bin_str_reversed = np.array(bin_str_reversed)

    # Check if optimal path in a subset of most probable paths:
    sorted_list_of_mostprobable_paths = sorted(
        dist_binary_prob, key=dist_binary_prob.get, reverse=True
    )

    # Dictionary keys and values where key = binary path, value = probability:
    # Find maximal probability in all values of the dictionary:
    max_probability = max(dist_binary_prob.values())
    selected_paths = []
    for path, probability in dist_binary_prob.items():

        probability = probability / max_probability
        dist_binary_prob[path] = probability
        # print(
        #     f"Path (quantum read -> right=q0): {path} with ratio proba/max_proba : {probability}"
        # )

        percentage = 0.5
        # Select paths with probability higher than percentage of the maximal probability:
        if probability > percentage:
            selected_paths.extend([path, probability])
    # Sort the selected paths by probability from most probable to least probable:

    selected_paths = sorted(
        selected_paths[::2], key=lambda x: dist_binary_prob[x], reverse=True
    )

    # print("_______________________________________________________________________\n")
    # print(
    #     f"Selected paths among {percentage*100} % of solutions (right=q0) from most probable to least probable: {selected_paths}"
    # )

    # print(
    #     f"Optimal path obtained by diagonal hamiltonian minimum costs (right=q0): {h.exact_path}"
    # )

    # match_found = False
    # for i in selected_paths:
    #     if i in h.exact_path:
    #         match_found = True
    #         break

    plot_distribution(
        {key: dist_binary_prob[key] for key in selected_paths},
        figsize=(16, 14),
        title=("Distribution of probabilities for selected paths"),
        # \n Right path FOUND (quantum read): {h.exact_path}"
        # if match_found
        # else f"Distribution of probabilities for selected paths \n Right path NOT FOUND (quantum read): {h.exact_path}"
        color="pink",  # if match_found else "lightblue",
        sort="value_desc",
        filename=visu_out_file_selected,
    )
    print("Distribution of probabilities for selected paths saved in ", visu_out_file_selected)
    if not save_only:
        plt.show()
    # target_string=h.exact_path,)
    # if match_found:
    #     print(
    #         "The optimal solution is in the subset of solutions found by QAOA.\n______")

    # else:
    #     print(
    #         "The solution is not in given subset of solutions found by QAOA.\n_________________")
    #     )
