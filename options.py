import argparse

import consts
import params


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
        "--inst_parallel",
        type=int,
        default=0,
        help="Flag to control solving instances in parallel"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers. Used when inst_parallel is 1"
    )

    parser.add_argument(
        "--timelimit",
        type=int,
        default=600,
        help="Solver timelimit in seconds"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/setcover/data/1000_1000/",
        help="Folder containing lp files of training instances",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default=consts.HOMOGENEOUS,
        help="Type of the dataset. It can be HOMOGENEOUS or HETEROGENEOUS."
    )

    parser.add_argument(
        "--strategy",
        help="Branching strategy for solving mip",
        type=int,
        default=consts.BS_PC
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
        default=consts.BETA
    )

    parser.add_argument(
        "--instance",
        help="Path to instance lp file",
        type=str,
        default="./data/setcover/data/1000_1000/instance_1.lp"
    )
    # default = "/scratch/rahulpat/setcover/train/1000_1000/1000_1000_0.lp"

    parser.add_argument(
        "--seed",
        help="Seed for CPLEX",
        type=int,
        default=3
    )
    opts = parser.parse_args(args)

    return opts
