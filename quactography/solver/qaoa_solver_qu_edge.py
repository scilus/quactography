import multiprocessing
import itertools
import sys
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r"C:\Users\harsh\quactography")

from quactography.solver.io import save_optimization_results
from quactography.solver.optimization_loops import (
    POWELL_loop_optimizer,
    POWELL_refinement_optimization,
)

alpha_min_costs = []


# Minimization cost function
def cost_func(params, estimator, ansatz, hamiltonian):
    cost = (
        estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    )
    return cost


# Function to find the shortest path in a graph using QAOA algorithm with parallel processing:
def find_longest_path(args):
    """Summary :  Usage of QAOA algorithm to find the shortest path in a graph.

    Args:
        h (Sparse Pauli list):  Hamiltonian in QUBO representation
        reps (int): number of layers to add to quantum circuit (p layers)
        outfile ( str) : name of output file to save optimization results

    Returns:
        res (minimize):  Results of the minimization
        min_cost (float):  Minimum cost
        alpha_min_cost (list):  List of alpha, minimum cost and binary path
    """
    h = args[0]
    reps = args[1]
    outfile = args[2]
    optimizer = args[3]
    num_refinement_loops = args[4]
    epsilon = args[5]
    # Save output file name diffrerent for each alpha:
    outfile = outfile + "_alpha_" + str(h.alpha)

    # # Pad with zeros to the left to have the same length as the number of edges:
    # for i in range(len(h.exact_path[0])):
    #     if len(h.exact_path[0]) < h.graph.number_of_edges:
    #         h.exact_path[i] = h.exact_path[i].zfill(h.graph.number_of_edges + 1)
    # # print("Path Hamiltonian (quantum reading -> right=q0) : ", h.exact_path)

    # # Reverse the binary path to have the same orientation as the classical path:
    # h.exact_path_classical_read = [path[::-1] for path in h.exact_path]

    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h.total_hamiltonian, reps, name="QAOA")

    # Plot the circuit layout:
    ansatz.decompose(reps=3).draw(output="mpl", style="iqp")

    # ----------------------------------------------------------------RUN LOCALLY: --------------------------------------------------------------------------------------
    # Run on local estimator and sampler:
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    sampler = Sampler(options={"shots": 1000000, "seed": 42})
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------

    if optimizer == "Powell":
        # Initialize list of invalid parameters
        no_valid_params = []

        # Initialize parameters for the optimizer
        x_0 = np.zeros(ansatz.num_parameters)
        epsilon = epsilon
        previous_cost = np.inf
        cost_history = []
        loop_count = 0
        max_loops = 50
        print(
            f"Using Powell optimizer with {num_refinement_loops} refinement loops, epsilon = {epsilon}, and max_loops = {max_loops}"
        )
        # Run initial optimization loop
        res, last_cost, previous_cost, x_0, loop_count, cost_history = (
            POWELL_loop_optimizer(
                loop_count,
                max_loops,
                previous_cost,
                epsilon,
                x_0,
                cost_history,
                estimator,
                ansatz,
                h,
            )
        )
        if num_refinement_loops > 0:
            res, last_cost, previous_cost, x_0, loop_count, cost_history = (
                POWELL_refinement_optimization(
                    loop_count,
                    max_loops,
                    estimator,
                    ansatz,
                    h,
                    no_valid_params,
                    epsilon,
                    x_0,
                    previous_cost,
                    last_cost,
                    cost_history,
                    num_refinement_loops,
                )
            )  # type: ignore

    # if optimizer == "SPSA": TODO: Implement SPSA optimizer

    # Save the minimum cost and the corresponding parameters
    min_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)  # type: ignore
    print("parameters after optimization loop : ", res.x, "Cost:", min_cost)  # type: ignore

    # Plot cost function:
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label="Cost evolution")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.title("Convergence of cost during optimisation")
    plt.legend()
    plt.grid(True)
    plt.savefig("cost_history_plot")

    circ = ansatz.copy()
    circ.measure_all()
    dist = sampler.run(circ, res.x).result().quasi_dists[0]  # type: ignore
    dist_binary_probabilities = dist.binary_probabilities()

    bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
    bin_str_reversed = bin_str[::-1]
    bin_str_reversed = np.array(bin_str_reversed)  # type: ignore

    # Concatenate the binary path to a string:
    str_path_reversed = ["".join(map(str, bin_str_reversed))]  # type: ignore
    str_path_reversed = str_path_reversed[0]  # type: ignore

    # Save parameters alpha and min_cost with path in csv file:
    opt_path = str_path_reversed

    save_optimization_results(
        dist=dist,
        dist_binary_probabilities=dist_binary_probabilities,
        min_cost=min_cost,
        hamiltonian=h,
        outfile=outfile,
        opt_bin_str=opt_path,
        reps=reps,
        opt_params=res.x,  # type: ignore
    )  # type: ignore


def multiprocess_qaoa_solver_edge(
    hamiltonians,
    reps,
    nbr_processes,
    output_file,
    optimizer,
    number_refine_loops,
    epsilon,
):
    pool = multiprocessing.Pool(nbr_processes)

    results = pool.map(
        find_longest_path,
        zip(
            hamiltonians,
            itertools.repeat(reps),
            itertools.repeat(output_file),
            itertools.repeat(optimizer),  # type: ignore
            itertools.repeat(number_refine_loops),
            itertools.repeat(epsilon),
        ),
    )
    pool.close()
    pool.join()

    print(
        "------------------------MULTIPROCESS SOLVER FINISHED-------------------------------"
    )
