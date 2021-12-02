import os
from pathlib import Path

import consts
from utils.cplex_model import apply_branch_history
from utils.cplex_model import create_default_branches
from utils.cplex_model import disable_output
from utils.cplex_model import find_cutoff
from utils.cplex_model import get_branch_solution
from utils.cplex_model import get_candidates
from utils.cplex_model import get_clone
from utils.cplex_model import get_data
from utils.cplex_model import get_logging_callback
from utils.cplex_model import get_sb_scores
from utils.cplex_model import save_mip_solve_info
from utils.cplex_model import set_params
from utils.cplex_model import solve_as_lp
from utils.cplex_model import solve_branching
from utils.cplex_model import update_meta_model_param


def get_paths(opts):
    # Generate paths
    instance_paths, output_path = None, None
    output_path = Path(opts.output)

    # An instance gets preference over a dataset if it is explicitly provided
    if opts.instance is None:
        data_path = Path(opts.dataset)
        assert data_path.is_dir(), "Dataset path not a directory!"
        # Generate instance paths
        if opts.parallel == 1:
            all_instance_paths = [p for p in data_path.iterdir()]
            all_instance_paths.sort()
            dataset_size = len(all_instance_paths)

            # Total number of slurm workers detected
            # Defaults to 1 if not running under SLURM
            N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

            # This worker's array index. Assumes slurm array job is zero-indexed
            # Defaults to zero if not running under SLURM
            this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

            instance_paths = [all_instance_paths[param_idx]
                              for param_idx in range(this_worker, dataset_size, N_WORKERS)]
        elif opts.warm_start != consts.NONE:
            instance_paths = [p for p in data_path.iterdir()]
            instance_paths.sort()
    else:
        # Explicit instance path provided
        instance_paths = [Path(opts.instance)]

    return instance_paths, output_path
