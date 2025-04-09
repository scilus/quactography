import os
import numpy as np


def save_optimization_results(
    outfile,
    outfolder,
    dist,
    min_cost,
    hamiltonian,
    dist_binary_probabilities,
    opt_bin_str,
    reps,
    opt_params,
):
    """
    Save the optimization results to a file in .npz format.

    Parameters
    ----------
    outfile : str
        The output file name for the optimization results in .npz format.
    dist : float
        The distribution of the optimal path.
    min_cost : float
        The minimum cost of the optimization.
    hamiltonian : Hamiltonian object from quactography library
        The Hamiltonian used for the optimization.
    dist_binary_probabilities : dict
        Dictionary containing the binary probabilities of the paths.
    opt_bin_str : str
        The optimal binary string of the path.
    reps : int
        The number of repetitions of the optimization, determines the number of sets of gamma and beta angles.
    opt_params : list of floats
        The optimal parameters for the quantum circuit.
    Returns
    -------
    None
    """
    if os.path.exists(outfolder) == False:
        os.makedirs(outfolder)
        
    np.savez(
        outfolder+outfile,
        dist=dist,
        dist_binary_probabilities=dist_binary_probabilities,
        min_cost=min_cost,
        hamiltonian=hamiltonian,
        opt_bin_str=opt_bin_str,
        reps=reps,
        opt_params=opt_params,
    )


def load_optimization_results(in_file):
    """
    Load the optimization results from a file in .npz format.

    Parameters
    ----------
    in_file : str
        The input file containing the optimization results in .npz format.
    Returns
    -------
    dist : float
        The distance of the optimal path.
    dist_binary_probabilities : dict
        Dictionary containing the binary probabilities of the paths.
    min_cost : float
        The minimum cost of the optimization.
    hamiltonian : Hamiltonian object from quactography library
        The Hamiltonian used for the optimization. 
    opt_bin_str : str
        The optimal binary string of the path.
    reps : int
        The number of repetitions of the optimization,
        determines the number of sets of gamma and beta angles.
    opt_params : list of floats
        The optimal parameters for the quantum circuit.
    """
    data = np.load(in_file, allow_pickle=True)
    return (
        data["dist"],
        data["dist_binary_probabilities"],
        data["min_cost"],
        data["hamiltonian"],
        data["opt_bin_str"],
        data["reps"],
        data["opt_params"],
    )
