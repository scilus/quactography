import multiprocessing
import itertools

# import sys
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from functools import partial
from functools import partial

# sys.path.append(r"C:\Users\harsh\quactography")

from quactography.solver.io import save_optimization_results
from quactography.solver.optimization_loops import (
    POWELL_loop_optimizer,
    POWELL_refinement_optimization,
)
from quactography.visu.plot_cost_landscape import plt_cost_func

# !!!!!!!!!!! Optimal path returned from this optimisation is in
# classical read (meaning the zeroth qubit is the first from left
# term in binary string) !!!!!!!!!!!!!!!

alpha_min_costs = []


# Minimization cost function
def cost_func(params, estimator, ansatz, hamiltonian):
    """
    Cost function to minimize for the optimization of the quantum circuit.

    Parameters
    ----------
    params : list
        List of parameters for the quantum circuit.
        (gamma, beta) angles depending on the number of layers.
    estimator : Estimator from qiskit
        Estimator used to evaluate the cost function.
    ansatz : QuantumCircuit object from qiskit
        Quantum circuit used to generate the ansatz.
    hamiltonian : PauliSum object from qiskit
        Hamiltonian to minimize. Cost function in quantum Formalism.
    Returns
    -------
    cost : float
        Value of the cost function for the given parameters.
    """

    cost = (
        estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    )
    return cost


# Function to find the shortest path in a graph using QAOA algorithm with parallel processing:
def find_longest_path(args):
    """
    Find the longest path in a graph using the QAOA algorithm,
    with a plot of the cost landscape if reps=1 and
    the optimal point if cost_landscape=True.

    Parameters
    ----------
    args : tuple
        Tuple containing the Hamiltonian object from quactography library,
        Hamiltonian_qubit_edge, the number of repetitions for the QAOA algorithm,
        the output file name for the optimization results in .npz format, the optimizer
        to use for the QAOA algorithm, a boolean to plot the cost landscape with
        the optimal point if reps=1, and a boolean to save the figure
        without displaying it.


    Returns
    -------
    None
    """
    h = args[0]
    reps = args[1]
    outfile = args[2]
    optimizer = args[3]

    # Save output file name diffrerent for each alpha:
    outfile = outfile + "_alpha_" + str(h.alpha)

    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h.total_hamiltonian, reps, name="QAOA")

    # Plot the circuit layout:
    # ansatz.decompose(reps=3).draw()

    # ----------------------------------------------------------------RUN LOCALLY: -----
    # Run on local estimator and sampler:
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    sampler = Sampler(options={"shots": 1000000, "seed": 42})
    # -----------------------------------------------------------------------------------

    if optimizer == "Differential":
        # Reference: https://www.youtube.com/watch?v=o-OPrQmS1pU
        # Define fixed arguments
        cost_func_with_args = partial(
            cost_func,
            estimator=estimator,
            ansatz=ansatz,
            hamiltonian=h.total_hamiltonian,
        )

        # Call differential evolution with the modified cost function
        bounds = [[0, 2 * np.pi], [0, np.pi]] * reps
        res = differential_evolution(cost_func_with_args, bounds, disp=False)
        resx = res.x

    # Save the minimum cost and the corresponding parameters
    min_cost = cost_func(resx, estimator, ansatz, h.total_hamiltonian)  # type: ignore
    print("parameters after optimization loop : ", resx, "Cost:", min_cost)  # type: ignore

    # Scatter optimal point on cost Landscape ----------------------------
    if args[4]:
        if reps == 1:
            fig, ax1, ax2 = plt_cost_func(estimator, ansatz, h)
            ax1.scatter(  # type: ignore
                resx[0], resx[1], min_cost, color="red",
                marker="o", s=100, label="Optimal Point"  # type: ignore
            )
            ax2.scatter(  # type: ignore
                resx[0], resx[1], s=100, color="red",
                marker="o", label="Optimal Point"  # type: ignore
            )
            plt.savefig("Opt_point_visu.png")
            print("Optimal point saved in Opt_point_visu.png")
            if not args[5]:
                plt.show()
    else:
        pass
    # -----------------------------------------------------

    circ = ansatz.copy()
    circ.measure_all()
    dist = sampler.run(circ, resx).result().quasi_dists[0]  # type: ignore
    dist_binary_probabilities = dist.binary_probabilities()

    bin_str = list(map(int, max(dist.binary_probabilities(),
                                key=dist.binary_probabilities().get)))  # type: ignore
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
        # It is reversed as classical read to be compared to exact_path code when diagonalising Hamiltonian
        opt_bin_str=opt_path,
        reps=reps,
        opt_params=resx  # type: ignore
    )  # type: ignore


def multiprocess_qaoa_solver_edge(
    hamiltonians,
    reps,
    nbr_processes,
    output_file,
    optimizer,
    cost_landscape,
    save_only,
):
    """
    Solve the optimization problem using the QAOA algorithm
    with multiprocessing on the alpha values.

    Parameters
    ----------
    hamiltonians : list
        List of Hamiltonian objects from quactography library,
        Hamiltonian_qubit_edge.
    reps : int
        Number of repetitions for the QAOA algorithm,
        determines the number of sets of gamma and beta angles.
    nbr_processes : int
        Number of cpu to use for multiprocessing. default=1
    output_file : str
        The output file name for the optimization results in .npz format.
    optimizer : str
        Optimizer to use for the QAOA algorithm. default="Differential"
    cost_landscape : bool
        Plot the cost landscape with the optimal point if reps=1. default=False
    save_only : bool
        If True, the figure is saved without displaying it. default=False

    Returns
    -------
    None
    """
    pool = multiprocessing.Pool(nbr_processes)

    pool.map(
        find_longest_path,
        zip(
            hamiltonians,
            itertools.repeat(reps),
            itertools.repeat(output_file),
            itertools.repeat(optimizer),
            itertools.repeat(cost_landscape),
            itertools.repeat(save_only),
        ),
    )
    pool.close()
    pool.join()

    print(
        "------------------MULTIPROCESS SOLVER FINISHED-------------------------"
    )
