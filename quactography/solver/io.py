import numpy as np


def save_optimization_results(
    outfile, dist, min_cost, hamiltonian, dist_binary_probabilities, opt_bin_str, reps
):
    np.savez(
        outfile,
        dist=dist,
        dist_binary_probabilities=dist_binary_probabilities,
        min_cost=min_cost,
        hamiltonian=hamiltonian,
        opt_bin_str=opt_bin_str,
        reps=reps,
    )


def load_optimization_results(in_file):
    data = np.load(in_file, allow_pickle=True)
    return (
        data["dist"],
        data["dist_binary_probabilities"],
        data["min_cost"],
        data["hamiltonian"],
        data["opt_bin_str"],
        data["reps"],
    )
