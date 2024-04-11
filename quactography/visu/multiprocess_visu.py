import multiprocessing
import itertools
from quactography.visu.dist_prob import _plot_distribution_of_probabilities


def multiprocess_visu(
    in_files, nbr_processes, visu_out_file_total, visu_out_file_selected, h
):
    pool = multiprocessing.Pool(nbr_processes)

    pool.map(
        _plot_distribution_of_probabilities,  # type: ignore
        zip(
            in_files,
            itertools.repeat(visu_out_file_total),
            itertools.repeat(visu_out_file_selected),
            h,
        ),
    )
    pool.close()
    pool.join()

    print(
        "------------------------MULTIPROCESS VISU FINISHED-------------------------------"
    )
