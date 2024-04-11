import multiprocessing
import itertools

from quactography.solver.qaoa_solver import _find_longest_path

alpha_min_costs = []


def multiprocess_qaoa_solver(hamiltonians, reps, nbr_processes, output_file):
    pool = multiprocessing.Pool(nbr_processes)

    results = pool.map(
        _find_longest_path,
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
    return results
