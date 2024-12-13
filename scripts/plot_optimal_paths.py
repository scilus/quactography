import argparse

from quactography.visu.optimal_paths_edge import visualize_optimal_paths_edge

# from quactography.visu.optimal_paths_node import visualize_optimal_paths_node


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("mat_adj", help="Adjacency matrix", type=str)
    p.add_argument(
        "input_files",
        nargs="+",
        help="List of input files to plot optimal paths",
    )
    p.add_argument(
        "output_file",
        help="Output file name for visualisation",
        type=str,
    )

    return p


def main():
    """
    Plots of the resulting paths found and information on wether or not it is the right path that has been found or not.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    for i in range(len(args.input_files)):
        visualize_optimal_paths_edge(
            args.mat_adj + ".npz",
            args.input_files[i],
            args.output_file + "_" + str(i),
        )


if __name__ == "__main__":
    main()
