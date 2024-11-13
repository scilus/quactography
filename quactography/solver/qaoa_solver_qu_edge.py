import multiprocessing
import itertools
from math import floor
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

    # Minimization cost function
    def cost_func(params, estimator, ansatz, hamiltonian):
        cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=params)
            .result()
            .values[0]
        )
        return cost

    # Initialize list of invalid parameters
    no_valid_params = []

    # Optimizer function
    def loop_optimizer(
        loop_count, max_loops, previous_cost, epsilon, x_0, cost_history
    ):
        while loop_count < max_loops:
            print("Loop:", loop_count)

            # Minimization with COBYLA or other method
            res = minimize(
                cost_func,
                x_0,
                args=(estimator, ansatz, h.total_hamiltonian),
                method="Powell",
                options={"maxiter": 500, "disp": False},
                tol=1e-5,
            )

            # Optimized cost
            new_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)
            print("Loop:", loop_count, "Iterations:", res.nfev, "Cost:", new_cost)

            # Check for convergence
            if abs(previous_cost - new_cost) < epsilon:
                print("Cost when convergence reached:", new_cost)

            # Update starting point and previous cost if improvement found
            if new_cost < previous_cost:
                no_valid_params.append(res.x)
                x_0 = res.x + 0.4
                previous_cost = new_cost
                cost_history.append(new_cost)
            else:
                no_valid_params.append(res.x)
                cost_history.append(new_cost)
                break

            if loop_count == max_loops - 1:
                print("Max loops reached")
                print("Cost when max loops reached:", new_cost)
                break

            loop_count += 1

        print("List of non-valid parameters:", no_valid_params)
        return res, new_cost, previous_cost, x_0, loop_count, cost_history

    # Initialize parameters for the optimizer
    epsilon = 1e-6
    x_0 = np.zeros(ansatz.num_parameters)
    previous_cost = np.inf
    cost_history = []
    loop_count = 0
    max_loops = 30

    # Run initial optimization loop
    res, last_cost, previous_cost, x_0, loop_count, cost_history = loop_optimizer(
        loop_count, max_loops, previous_cost, epsilon, x_0, cost_history
    )

    # Helper function to check if a parameter is close to any of the non valid parameters
    def is_close_to_any(x_0, no_valid_params, tol=1e-2):
        return any(np.allclose(x_0, param, atol=tol) for param in no_valid_params)

    # Main refinement loop
    num_refinement_loops = 5
    while last_cost > h.graph.min_weight and num_refinement_loops > 0:
        # Set x_0 to random parameters not in the list of non valid parameters
        while True:
            x_0 = np.random.uniform(0, np.pi, ansatz.num_parameters)
            if not is_close_to_any(x_0, no_valid_params):
                break

        # Run the optimizer
        (
            potential_res,
            potential_last_cost,
            potential_previous_cost,
            potential_x_0,
            loop_count,
            potential_cost_history,
        ) = loop_optimizer(
            loop_count, max_loops, previous_cost, epsilon, x_0, cost_history
        )
        # If cost is growing, cancel the refinement loop, return the last result
        if potential_last_cost > last_cost:
            break
        if potential_last_cost < last_cost:
            last_cost = potential_last_cost
            res = potential_res
            previous_cost = potential_previous_cost
            x_0 = potential_x_0
            cost_history = potential_cost_history

        num_refinement_loops -= 1

    # Save the minimum cost and the corresponding parameters
    min_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)
    print("parameters after optimization loop : ", res.x, "Cost:", min_cost)

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

    save_optimization_results(
        dist=dist,
        dist_binary_probabilities=dist_binary_probabilities,
        min_cost=min_cost,
        hamiltonian=h,
        outfile=outfile,
        opt_bin_str=opt_path,
        reps=reps,
        opt_params=res.x,
    )  # type: ignore


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
