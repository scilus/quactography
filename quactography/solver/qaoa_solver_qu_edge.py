import multiprocessing
import itertools
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


from quactography.solver.io import save_optimization_results


alpha_min_costs = []


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

    # Define a small value as accepted difference to cease optimisation process:
    epsilon = 1e-6

    # Initialise parameters to zeros:
    x_0 = np.zeros(ansatz.num_parameters)

    # previous cost initialise to a very large number:
    previous_cost = np.inf
    cost_history = []

    # Minimisation cost function:
    def cost_func(params, estimator, ansatz, hamiltonian):
        cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=params)
            .result()
            .values[0]
        )
        return cost

    iteration = 0
    # Optimisation loop
    while True:
        # With Cobyla, or other optimizer
        res = minimize(
            cost_func,
            x_0,
            args=(estimator, ansatz, h.total_hamiltonian),
            method="COBYLA",
            options={"maxiter": 5000, "disp": False},
            tol=1e-4,
        )
        print(res)
        # Optimised cost:
        new_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)
        cost_history.append(new_cost)
        print(f"Iteration {iteration}: Cost = {new_cost}")
        iteration += 1

        # Verify distance between previous cost and new one:
        if (abs(previous_cost - new_cost)) ** 2 < epsilon:
            opt_params = res.x
            break

        # Break if more than 1000 iterations :
        if iteration > 1000:
            opt_params = res.x
            print("maximum number of iterations attained!!")
            break

        # If new cost better, x_0 found is updated to the result found :
        if new_cost < previous_cost:
            x_0 = res.x
            previous_cost = new_cost

    print("parameters after optimization loop : ", ansatz.parameters)
    # Plot cost function:
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label="Cost evolution")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.title("Convergence of cost during optimisation")
    plt.legend()
    plt.grid(True)
    plt.savefig("cost_history_plot")

    # # Old way of minimising :
    # # Cost function for the minimizer:--------------------------------------------------------------
    # def cost_func(params, estimator, ansatz, hamiltonian):
    #     cost = (
    #         estimator.run(ansatz, hamiltonian, parameter_values=params)
    #         .result()
    #         .values[0]
    #     )
    #     return cost

    # x0 = np.zeros(ansatz.num_parameters)

    # # Minimize the cost function using COBYLA method:
    # res = minimize(
    #     cost_func,
    #     x0,
    #     args=(estimator, ansatz, h.total_hamiltonian),
    #     method="COBYLA",
    #     options={"maxiter": 5000, "disp": False},
    #     tol=1e-4,
    # )
    # ----------------------------------------------------------------------------------------------------

    min_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)
    circ = ansatz.copy()
    circ.measure_all()
    dist = sampler.run(circ, res.x).result().quasi_dists[0]
    dist_binary_probabilities = dist.binary_probabilities()

    bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
    bin_str_reversed = bin_str[::-1]
    bin_str_reversed = np.array(bin_str_reversed)  # type: ignore

    # Concatenate the binary path to a string:
    str_path_reversed = ["".join(map(str, bin_str_reversed))]  # type: ignore
    str_path_reversed = str_path_reversed[0]  # type: ignore

    # Save parameters alpha and min_cost with path in csv file:
    opt_path = str_path_reversed
    print("opt_params", opt_params)
    save_optimization_results(dist=dist, dist_binary_probabilities=dist_binary_probabilities, min_cost=min_cost, hamiltonian=h, outfile=outfile, opt_bin_str=opt_path, reps=reps, opt_params=opt_params)  # type: ignore


def multiprocess_qaoa_solver_edge(hamiltonians, reps, nbr_processes, output_file):
    pool = multiprocessing.Pool(nbr_processes)

    results = pool.map(
        find_longest_path,
        zip(
            hamiltonians,
            itertools.repeat(reps),
            itertools.repeat(output_file),
        ),
    )
    pool.close()
    pool.join()

    print(
        "------------------------MULTIPROCESS SOLVER FINISHED-------------------------------"
    )
