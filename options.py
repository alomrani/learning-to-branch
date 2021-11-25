import argparse
import os
import time

import torch

import consts


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Options for learning to branch"
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=consts.BRANCHING,
        help="Generate optimal solution or do branching"
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

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=1000.,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="datasets/val.pt",
        help="Dataset file to use for validation",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="datasets/test.pt",
        help="Dataset file to use for testing",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="./data/train",
        help="Folder containing lp files of training instances",
    )

    parser.add_argument(
        "--dataset_size", type=int, default=10000, help="Dataset size for training",
    )
    parser.add_argument(
        "--lr_model",
        type=float,
        default=0.001,
        help="Set the learning rate for the model",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch"
    )
    # Misc
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Set this to true if you want to tune the hyperparameters",
    )

    parser.add_argument(
        "--output_dir", default="./outputs", help="Directory to write model outputs to"
    )

    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=0,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )

    parser.add_argument(
        "--save_dir", help="Path to save the checkpoints",
    )

    parser.add_argument(
        "--strategy",
        help="Branching strategy for solving mip",
        type=int,
        default=consts.BS_PC
    )

    parser.add_argument(
        "--instance",
        help="Path to instance lp file",
        type=str,
        default="/scratch/rahulpat/setcover/train/1000_1000/1000_1000_0.lp"
    )

    parser.add_argument(
        "--seed",
        help="Seed for CPLEX",
        type=int,
        default=3
    )

    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.use_cuda:
        opts.device = "cuda"
    else:
        opts.device = "cpu"

    opts.run_name = "{}_{}".format("run", time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(opts.output_dir, opts.run_name)
    assert (
            opts.dataset_size % opts.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"

    return opts
