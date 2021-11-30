import os
import pickle as pkl
from pathlib import Path

import consts
from utils.cplex_model import apply_branch_history
from utils.cplex_model import disable_output
from utils.cplex_model import get_branch_solution
from utils.cplex_model import get_candidates
from utils.cplex_model import get_clone
from utils.cplex_model import get_data
from utils.cplex_model import get_logging_callback
from utils.cplex_model import get_sb_scores
from utils.cplex_model import set_params
from utils.cplex_model import solve_as_lp


def get_paths(opts):
    # HOMOGENEOUS: /scratch/rahulpat/l2b/setcover/data/1000_1000/
    # HETEROGENEOUS: /scratch/rahulpat/l2b/benchmark/data/
    # Generate data path
    data_path = Path(opts.dataset)
    assert data_path.is_dir(), "Dataset path not a directory!"

    # Generate output path
    if opts.dataset_type == consts.HOMOGENEOUS:
        # /scratch/rahulpat/l2b/setcover/output/1000_1000/
        output_path = data_path.parent.parent / "output" / data_path.name
    else:
        # /scratch/rahulpat/l2b/benchmark/output/
        output_path = data_path.parent / "output"

    # Generate instance paths
    if opts.inst_parallel == 1:
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
        instance_paths = [Path(opts.instance)]

    return data_path, output_path, instance_paths


def get_optimal_obj_dict(output_path, instance_path):
    # Check if optimal solution exists to provide as primal bound
    primal_bound = 1e6
    opt_dict = None
    optimal_obj_path = output_path.joinpath(f"optimal_obj/{instance_path.stem}.pkl")
    print(f"* Checking optimal objective pickle at {optimal_obj_path}...")
    if optimal_obj_path.exists():
        opt_dict = pkl.load(open(optimal_obj_path, 'rb'))
        if instance_path.name in opt_dict and opt_dict[instance_path.name] is not None:
            primal_bound = opt_dict[instance_path.name]
            print(f"\t** Primal bound: {primal_bound}")
        else:
            print(f"\t** Instance primal bound not found...")
    else:
        print("\t** Warning: Optimal objective pickle not found. Can't set primal bound.")

    return opt_dict, primal_bound
