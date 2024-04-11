import argparse

from quactography.visu.optimal_paths import visualize_optimal_paths


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
    parser = _build_arg_parser()
    args = parser.parse_args()
    for i in range(len(args.input_files)):
        visualize_optimal_paths(
            args.mat_adj,
            args.input_files[i],
            args.output_file + "_" + str(i),
        )


if __name__ == "__main__":
    main()
