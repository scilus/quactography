import argparse

from quactography.visu.dist_prob_edge import plot_distribution_of_probabilities_edge

    """Tool to visualize 10% of most optimal paths found by QAOA, and all the histogram of paths as well in a second plot.  
    """
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "input_files",
        nargs="+",
        help="List of input files to plot distribution (add .npz extension)",
    )
    p.add_argument(
        "visual_dist_output_file_total",
        help="Output file name for visualisation",
        type=str,
    )
    p.add_argument(
        "visual_dist_output_file_selected",
        help="Output file name for visualisation",
        type=str,
    )

    return p


def main():
    """
    Plots histogram of results for a selected pool and also for every solutions found. Color pink if right path is found, blue elsewise
    using matplotlib.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    for i in range(len(args.input_files)):
        plot_distribution_of_probabilities_edge(
            args.input_files[i],
            args.visual_dist_output_file_total + "_" + str(i),
            args.visual_dist_output_file_selected + "_" + str(i),
        )


if __name__ == "__main__":
    main()