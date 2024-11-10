import multiprocessing
import itertools
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
import numpy as np

from quactography.solver.io import save_optimization_results


alpha_min_costs = []


# Function to find the shortest path in a graph using QAOA algorithm with parallel processing:
def find_longest_path(args):
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

    # Run on local estimator and sampler:
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    sampler = Sampler(options={"shots": 1000000, "seed": 42})

    # Cost function for the minimizer:
    def cost_func(params, estimator, ansatz, hamiltonian):
        cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=params)
            .result()
            .values[0]
        )
        return cost

    x0 = np.zeros(ansatz.num_parameters)

    # Minimize the cost function using COBYLA method:
    res = minimize(
        cost_func,
        x0,
        args=(estimator, ansatz, h.total_hamiltonian),
        method="COBYLA",
        options={"maxiter": 5000, "disp": False},
        tol=1e-4,
    )

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

    save_optimization_results(dist=dist, dist_binary_probabilities=dist_binary_probabilities, min_cost=min_cost, hamiltonian=h, outfile=outfile, opt_bin_str=opt_path, reps=reps)  # type: ignore


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
