import multiprocessing
import itertools
import os
from pathlib import Path

from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import differential_evolution
from functools import partial

from quactography.solver.io import load_optimization_results, save_optimization_results
from quactography.solver.optimization_loops import (
    POWELL_loop_optimizer,
    POWELL_refinement_optimization,
)
from quactography.visu.plot_cost_landscape import plt_cost_func

# !!!!!!!!!!! Optimal path returned from this optimisation is in
# classical read (meaning the zeroth qubit is the first from left
# term in binary string) !!!!!!!!!!!!!!!


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
        estimator.run(ansatz, hamiltonian, parameter_values=params)
        .result().values[0]
    )
    return cost

def mixer(h):
    """
    Create the mixer operator for the QAOA algorithm.

    Returns
    -------
    mixer : SparsePauliOp object from qiskit
        Mixer operator for the QAOA algorithm.
    """
    
    
    pauli_weight_first_term = [
            ("I" * h.graph.number_of_edges, h.graph.all_weights_sum / 2)
        ]
    pauli_weight_first_term.append(("X" + "I" * (h.graph.number_of_edges-1),-h.graph.weights[0][1] / 2))
    # Z à la bonne position:
    for i in range(1, h.graph.number_of_edges):
        str1 = (
            "I" * (i-1) + "Y" + "X"  + "I" * (h.graph.number_of_edges - i - 1),-h.graph.weights[0][i] / 2
            
        )
        pauli_weight_first_term.append(str1)
    pauli_weight_first_term.append(("I" * (h.graph.number_of_edges-1) + "Y",-h.graph.weights[0][i] / 2))
    mixer = SparsePauliOp.from_list(pauli_weight_first_term)
        
    return mixer


# Function to find the shortest path in a graph
# using QAOA algorithm with parallel processing:
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
    outfileI = args[2]
    optimizer = args[3]

    mixerf = mixer(h)

    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h.total_hamiltonian, reps, mixer_operator=mixerf, name="QAOA", flatten=True)
    # Plot the circuit layout:
    # ansatz.decompose(reps=3).draw()

    # ----------------------------------------------------------------RUN LOCALLY: -----
    # Run on local estimator and sampler:
    # Save output file name diffrerent for each alpha and loop:
    outfile = outfileI + "_alpha_" + str(h.alpha_init) + "_reps_" + str(reps)
    estimator = Estimator(options={"shots": 1000000, "seed": 43})
    sampler = Sampler(options={"shots": 1000000, "seed": 43})
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
        outfolder=args[5],
        # It is reversed as classical read to be compared to exact_path code when diagonalising Hamiltonian
        opt_bin_str=opt_path,
        reps=reps,
        opt_params=resx  # type: ignore
    )  # type: ignore
    
def find_min_cost(in_folder, alpha, rep):
    """
    Find the minimum cost in the optimization results for given alpha and rep,
    and remove all files with a cost lower than the minimum cost.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    QUACK_RAP does not save any files, nor have iteration with different alphas or repetitions,
    it only returns a list of coordinates for the streamline.

    Parameters
    ----------
    in_folder : str
        The folder containing the optimization results files.
    alpha : float
        The alpha value used in the optimization.
    rep : int
        The number of repetitions for the QAOA algorithm.

    Returns
    -------
    None
    """
    path = Path(in_folder)
    glob_path = path.glob(f'*_alpha_{alpha}_reps_{rep}.npz')
    min = 100
    
    for in_file_path in glob_path:
        _, _, min_cost, _, _, _, _ = load_optimization_results(in_file_path)
        min_cost = min_cost.item()
        if min_cost < min:
            min = min_cost
        else:
            os.remove(in_file_path)
            


def multiprocess_qaoa_solver_edge(
    hamiltonians,
    reps,
    nbr_processes,
    output_file,
    output_folder,
    optimizer,
    cost_landscape,
    save_only
):
    """
    Solve the optimization problem using the QAOA algorithm
    with multiprocessing on the alpha values.

    Parameters
    ----------
    hamiltonians : list
        List of Hamiltonian objects from quactography library, Hamiltonian_qubit_edge.
    batch_count : int
        Number of time the command will be ran
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
            itertools.repeat(output_folder),
            itertools.repeat(save_only),
        ),
    )
    print(
        "------------------MULTIPROCESS SOLVER FINISHED-------------------------"
    )
