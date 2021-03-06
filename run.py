import sys

import numpy as np

import consts
from options import get_options
from utils import find_cutoff, get_paths, solve_branching, update_meta_model_param


def run(opts):
    print(f'* Run mode: {consts.MODE[opts.mode]}')
    np.random.seed(opts.seed)

    # Get paths to output and instances
    instance_paths, output_path = get_paths(opts)

    if opts.mode == consts.GENERATE_OPTIMAL:
        for instance_path in instance_paths:
            find_cutoff(instance_path, output_path, opts)

    # Training the meta-model must be done sequentially
    elif opts.mode == consts.TRAIN_META:
        print(f'* Warm-start strategy: {consts.WARM_START[opts.warm_start]}')
        print(f'* Beta: {opts.beta}, Theta: {opts.theta}')

        assert opts.beta > 0 and opts.theta > 0 and opts.theta2 > 0, "Beta, theta, theta2 must be set"
        assert opts.warm_start > 0, "Must set meta-model generation scheme"

        # Load relevant solve_instance()
        baseline_strategy = (
                opts.strategy == consts.CPX_DEFAULT
                or opts.strategy == consts.CPX_PC
                or opts.strategy == consts.CPX_SB
                or opts.strategy == consts.BS_SB
                or opts.strategy == consts.BS_PC
                or opts.strategy == consts.BS_SB_PC
        )

        meta_model_param, warm_start_model = None, None
        theta = opts.theta2
        num_instances_trained = 0
        for i, f in enumerate(instance_paths):
            # Only process instances that are solved by the CPLEX to
            # optimality and use their optimal objective value as cutoff
            if num_instances_trained >= opts.beta:
                break
            c, log_cb, vsel_cb = solve_branching(f, output_path, opts,
                                                 warm_start_model=warm_start_model)
            if c is None:
                continue
            if vsel_cb.times_called >= opts.theta2:
                trained_model = vsel_cb.model
                num_instances_trained += 1
                meta_model_param, warm_start_model = update_meta_model_param(meta_model_param, trained_model,
                                                                             num_instances_trained, opts)
        print(
            f"* Meta Model generated and saved at: pretrained/{f.parent.name}_{opts.beta}_{opts.theta2}_{consts.WARM_START[opts.warm_start]}.joblib")

    elif opts.mode == consts.BRANCHING:
        # Solve multiple instances in parallel using SLURM array jobs
        if opts.strategy > 5:
            assert opts.theta > 0, "Theta must be set"
        if opts.warm_start != 0:
            assert opts.beta > 0 and opts.theta > 0 and opts.theta2 > 0, "Beta, theta, theta2 must be set"

        assert 0 <= opts.strategy <= len(consts.STRATEGY), "Unknown branching strategy"
        for instance_path in instance_paths:
            c, log_cb, *_ = solve_branching(instance_path, output_path, opts)
        

if __name__ == "__main__":
    """
    * Usage instructions
    ----------------------------------------------------------------------------------------
    1. For generating mip cutoffs
    python run.py --mode 0 --instance <instance path> --seed <seed>
    
    2. For branching
    python run.py --mode 1 --instance <instance path> --strategy <strategy_id> --seed <seed>
    
    3. For learning meta-model
    
    * Parameters details
    ----------------------------------------------------------------------------------------     
    1. <strategy_id> can be between 0 to 5, where 
        0 ==> DEFAULT
        1 ==> Strong branching
        2 ==> Pseudocost branching
        3 ==> Strong(theta) + Pseudocost branching
        4 ==> Strong(theta) + SVM Rank
        5 ==> Strong(theta) + Feed forward Neural Network
        

    """
    opts = get_options(sys.argv[1:])
    run(opts)
