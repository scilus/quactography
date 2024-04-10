import numpy as np


def save_optimization_results(
    outfile, dist, min_cost, hamiltonian, dist_binary_probabilities
):
    np.savez(
        outfile,
        dist=dist,
        dist_binary_probabilities=dist_binary_probabilities,
        min_cost=min_cost,
        hamiltonian=hamiltonian,
    )


def load_optimization_results(in_file):
    data = np.load(in_file, allow_pickle=True)
    return (
        data["dist"],
        data["dist_binary_probabilities"],
        data["min_cost"],
        data["hamiltonian"],
    )
