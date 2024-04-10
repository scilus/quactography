from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
import numpy as np


# Function to find the shortest path in a graph using QAOA algorithm with parallel processing:
def _find_longest_path(args):
    """Summary :  Usage of QAOA algorithm to find the shortest path in a graph.

    Args:
        args (Sparse Pauli list):  Hamiltonian in QUBO representation

    Returns:
        res (minimize):  Results of the minimization
        min_cost (float):  Minimum cost
        alpha_min_cost (list):  List of alpha, minimum cost and binary path
    """
    h = args[0]
    reps = args[1]

    # Pad with zeros to the left to have the same length as the number of edges:
    for i in range(len(h.exact_path)):
        h.exact_path[i] = h.exact_path[i].zfill(len(h.starting_node_c) + 1)
    print("Path Hamiltonian (quantum reading -> right=q0) : ", h.exact_path)

    # Reverse the binary path to have the same orientation as the classical path:
    h.exact_path_classical_read = [path[::-1] for path in h.exact_path]

    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h.total_hamiltonian, reps, name="QAOA")

    # Plot the circuit layout:
    ansatz.decompose(reps=3).draw(output="mpl", style="iqp")

    # Run on local estimator and sampler. Fix seeds for results reproducibility.
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    sampler = Sampler(options={"shots": 1000000, "seed": 42})

    # Cost function for the minimizer.
    # Returns the expectation value of circuit with Hamiltonian as an observable.
    def cost_func(params, estimator, ansatz, hamiltonian):
        cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=params)
            .result()
            .values[0]
        )
        return cost

    x0 = np.zeros(ansatz.num_parameters)
    # Minimize the cost function using COBYLA method
    res = minimize(
        cost_func,
        x0,
        args=(estimator, ansatz, h.total_hamiltonian),
        method="COBYLA",
        # callback=callback,
        options={"maxiter": 5000, "disp": False},
        tol=1e-4,
    )

    # Close the progress bar once optimization is done
    # progress.close()

    min_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)

    # DIST OUTPUT:
    # # Get probability distribution associated with optimized parameters.
    # circ = ansatz.copy()
    # circ.measure_all()
    # dist = sampler.run(circ, res.x).result().quasi_dists[0]
    # # Plot distribution of probabilities:
    # plot_distribution(
    #     dist.binary_probabilities(),
    #     figsize=(10, 8),
    #     title="Distribution of probabilities",
    #     color="pink",
    # )
    # # Save plot of distribution:
    # # plt.savefig(f"output/distribution_alpha_{alpha:.2f}.png")

    # # print(max(dist.binary_probabilities(), key=dist.binary_probabilities().get))  # type: ignore
    # bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
    # bin_str_reversed = bin_str[::-1]
    # bin_str_reversed = np.array(bin_str_reversed)  # type: ignore

    # # Check if optimal path in a subset of most probable paths:
    # sorted_list_of_mostprobable_paths = sorted(dist.binary_probabilities(), key=dist.binary_probabilities().get, reverse=True)  # type: ignore

    # # Dictionary keys and values where key = binary path, value = probability:
    # # Find maximal probability in all values of the dictionary:
    # max_probability = max(dist.binary_probabilities().values())
    # selected_paths = []
    # for path, probability in dist.binary_probabilities().items():

    #     probability = probability / max_probability
    #     dist.binary_probabilities()[path] = probability
    #     # print(
    #     #     f"Path (quantum read -> right=q0): {path} with ratio proba/max_proba : {probability}"
    #     # )

    #     percentage = 0.5
    #     # Select paths with probability higher than percentage of the maximal probability:
    #     if probability > percentage:
    #         selected_paths.extend([path, probability])
    # # Sort the selected paths by probability from most probable to least probable:

    # selected_paths = sorted(
    #     selected_paths[::2], key=lambda x: dist.binary_probabilities()[x], reverse=True
    # )

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

    # plot_distribution(
    #     {key: dist.binary_probabilities()[key] for key in selected_paths},
    #     figsize=(16, 14),
    #     title=(
    #         f"Distribution of probabilities for selected paths \n Right path FOUND (quantum read): {h.exact_path}"
    #         if match_found
    #         else f"Distribution of probabilities for selected paths \n Right path NOT FOUND (quantum read): {h.exact_path}"
    #     ),
    #     color="pink" if match_found else "lightblue",
    #     sort="value_desc",
    #     filename=f"output/distribution_alpha_{h.alpha:.2f}.png",
    #     target_string=h.exact_path,
    # )
    # if match_found:
    #     print(
    #         "The optimal solution is in the subset of solutions found by QAOA.\n_______________________________________________________________________"
    #     )

    # else:
    #     print(
    #         "The solution is not in given subset of solutions found by QAOA.\n_______________________________________________________________________"
    #     )

    # # Concatenate the binary path to a string:
    # str_path_reversed = ["".join(map(str, bin_str_reversed))]  # type: ignore
    # str_path_reversed = str_path_reversed[0]  # type: ignore

    # # Save parameters alpha and min_cost with path in csv file:
    # alpha_min_cost = [h.alpha, min_cost, str_path_reversed]

    # # print(sorted(dist.binary_probabilities(), key=dist.bina
    # # ry_probabilities().get))  # type: ignore
    # print("Finished with alpha : ", h.alpha)

    return res, min_cost  # , alpha_min_cost, selected_paths
