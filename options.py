import argparse

import consts
import params


def set_default_path_str(opts):
    user = 'rahul'
    env = 'local'
    dataset_path_str, output_path_str = './data', './output'
    if user == 'rahul':
        if env == 'local':
            dataset_path_str = f'./data/setcover/data/1000_1000/'
            output_path_str = f'./data/setcover/output/1000_10001/'
        elif env == 'cc':
            dataset_path_str = '/scratch/rahulpat/l2b/setcover/data/1000_1000/'
            output_path_str = '/scratch/rahulpat/l2b/setcover/output/1000_1000/'
    elif user == 'md':
        # TODO: set relevant paths
        pass

    opts.dataset = dataset_path_str
    opts.output = output_path_str


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Options for learning to branch"
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=consts.BRANCHING,
        help="Generate optimal solution, train meta model or do branching"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Flag to control solving instances in parallel"
    )

    parser.add_argument(
        "--timelimit",
        type=int,
        default=200,
        help="Solver timelimit in seconds"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Folder containing lp files of training instances",
    )

    parser.add_argument(
        "--output",
        type=str,
        default='./',
        help="Folder to the dump the results"
    )

    parser.add_argument(
        "--strategy",
        help="Branching strategy for solving mip",
        type=int,
        default=consts.BS_PC
    )

    parser.add_argument(
        "--max_iterations",
        help="Maximum iterations for LP",
        type=int,
        default=50
    )

    parser.add_argument(
        "--theta",
        help="Number of data samples collected while training meta model",
        type=int,
        default=params.THETA
    )

    parser.add_argument(
        "--theta2",
        help="Number of data samples collected after warm-starting with meta model",
        type=int,
        default=params.THETA2
    )

    parser.add_argument(
        "--warm_start",
        help="warm_start setting: 0: no warm-start, 1: averaging, 2: incremental training",
        type=int,
        default=consts.NONE
    )

    parser.add_argument(
        "--beta",
        help="Number of instances used for training meta-model",
        type=int,
        default=params.BETA
    )

    parser.add_argument(
        "--instance",
        help="Path to instance lp file",
        type=str
    )
    # default = "/scratch/rahulpat/setcover/train/1000_1000/1000_1000_0.lp"

    parser.add_argument(
        "--seed",
        help="Seed for CPLEX",
        type=int,
        default=3
    )
    opts = parser.parse_args(args)

    if opts.dataset is None:
        set_default_path_str(opts)

    return opts
