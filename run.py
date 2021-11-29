import pickle as pkl
import sys
from pathlib import Path
import numpy as np
import consts
from options import get_options
from utils import disable_output
from models.MLPClassifier import MLPClassifier1 as MLPClassifier
import params
import cplex as CPX
import joblib
import os
from utils.cplex_model import get_candidates
from itertools import product

def update_meta_model_param(meta_model_param, new_model, iter, opts):

    warm_start_model = None
    if iter < opts.beta:
        # Incremental Averaging of weights
        if opts.warm_start == consts.AVERAGE_MODEL:
            if meta_model_param is not None:
                for i in range(len(new_model.coefs_)):
                    meta_model_param[0][i] += meta_model_param[0][i] + (new_model.coefs_[i] - meta_model_param[0][i]) / (iter + 1)
                    meta_model_param[1][i] += meta_model_param[1][i] + (new_model.intercepts_[i] - meta_model_param[1][i]) / (iter + 1)
            else:
                meta_model_param = (new_model.coefs_, new_model.intercepts_)
        elif opts.warm_start == consts.INCREMENTAL_WARM_START:
            warm_start_model = new_model

    if iter == opts.beta - 1:
        # Initialize meta model and save for future use
        if (opts.warm_start ==  consts.AVERAGE_MODEL and iter == opts.beta - 1):
            warm_start_model = MLPClassifier(verbose=True, init_params=meta_model_param, learning_rate_init=0.01, n_iter_no_change=30, max_iter=300, warm_start=True)

        joblib.dump(warm_start_model, f'pretrained/{opts.beta}_{opts.theta}_{consts.WARM_START[opts.warm_start]}.joblib')

    return meta_model_param, warm_start_model

def get_opt_dict(output_dir_path, instance_path):
    # Check if optimal solution exists to provide as primal bound
    primal_bound = 1e6
    opt_dict = None
    optimal_obj_path = output_dir_path.joinpath(f"optimal_obj/{instance_path.stem}.pkl")
    print(f"* Checking optimal objective pickle at {optimal_obj_path}...")
    if optimal_obj_path.exists():
        opt_dict = pkl.load(open(optimal_obj_path, 'rb'))
        if instance_path.name in opt_dict and opt_dict[instance_path.name] is not None:
            primal_bound = opt_dict[instance_path.name]
            print(f"* Primal bound: {primal_bound}")
        else:
            print(f"* Instance primal bound not found...")
    else:
        print("* Warning: Optimal objective pickle not found. Can't set primal bound.")
    
    return opt_dict, primal_bound
        

def save_solution(c, log_cb, instance_path, output_path):
    solve_status_id = c.solution.get_status()
    solve_status_verbose = c.solution.status[c.solution.get_status()]
    # Get the number of nodes and time taken to solve the instance
    num_nodes, total_time = None, None
    if solve_status_id == c.solution.status.MIP_optimal:
        num_nodes = c.solution.progress.get_num_nodes_processed()
        total_time = log_cb.total_time

    print(f"\n\tS: {solve_status_verbose}, T: {total_time}, N:{num_nodes}\n")

    # Prepare result
    result_dict = {str(instance_path): {'status': solve_status_id,
                                        'status_verbose': solve_status_verbose,
                                        'total_time': total_time,
                                        'num_nodes': num_nodes}}


    # Save results
    pkl.dump(result_dict, open(output_path, 'wb'))

def run(opts):

    print(f'* Run mode: {consts.MODE[opts.mode]}')

    # /scratch/rahulpat/setcover/data/1000_1000/
    data_path = Path(opts.dataset)
    # /scratch/rahulpat/setcover/output/1000_1000/
    output_dir_path = data_path.parent.parent / "output" / data_path.name

    if opts.mode == consts.GENERATE_OPTIMAL:
        if opts.inst_parallel == 1:
            # total number of slurm workers detected
            # defaults to 1 if not running under SLURM
            N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

            # this worker's array index. Assumes slurm array job is zero-indexed
            # defaults to zero if not running under SLURM
            this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

            for param_idx in range(this_worker, opts.dataset_size, N_WORKERS):
                instance_name = f"setcover_84_{data_path.name}_0.05_{param_idx}.lp"
                instance_path = data_path / instance_name

                solve_optimal(instance_path, output_dir_path, opts)
        else:
            solve_optimal(opts.instance, output_dir_path, opts)
        

    # Training the meta-model must be done sequentially
    elif opts.mode == consts.TRAIN_META:
        print(f'* Warm-start strategy: {consts.WARM_START[opts.warm_start]}')
        print(f'* Beta: {opts.beta}, Theta: {opts.theta}')

        assert opts.beta > 0 and opts.theta > 0 and opts.theta2 > 0, "Beta, theta, theta2 must be set"
        assert opts.warm_start > 0, "Must set meta-model generation scheme"

        # Load relevant solve_instance()
        baseline_strategy = (
                opts.strategy == consts.BS_DEFAULT
                or opts.strategy == consts.BS_SB
                or opts.strategy == consts.BS_PC
                or opts.strategy == consts.BS_SB_PC
        )
        if baseline_strategy:
            from strategy import baseline_solve_instance
            solve_instance = baseline_solve_instance
        else:
            from strategy import online_solve_instance
            solve_instance = online_solve_instance

        meta_model_param, warm_start_model = None, None
        theta = opts.theta
        for i, f in enumerate(data_path.glob('*.lp')):
            # Only process instances that are solved by the CPLEX to
            # optimality and use their optimal objective value as primal bound
            if i >= opts.beta:
                break
            opt_dict, primal_bound = get_opt_dict(output_dir_path, f)
                
            c, log_cb, trained_model = solve_instance(
                path=str(f),
                primal_bound=primal_bound,
                branch_strategy=opts.strategy,
                timelimit=opts.timelimit,
                seed=opts.seed,
                test=False,
                warm_start_model=warm_start_model,
                theta=theta
            )

            meta_model_param, warm_start_model = update_meta_model_param(meta_model_param, trained_model, i, opts)

            # /scratch/rahulpat/setcover/output/train/1000_1000/SB_PC
            beta_theta_dir = f"{opts.beta}_{opts.theta}_{opts.theta2}_{consts.WARM_START[opts.warm_start]}"
            output_dir_path1 = output_dir_path / consts.STRATEGY[opts.strategy] / beta_theta_dir
            output_dir_path1.mkdir(parents=True, exist_ok=True)
            # /scratch/rahulpat/setcover/output/train/1000_1000/SB_PC/1000_1000_0.pkl
            output_path = output_dir_path1.joinpath(str(f.stem) + ".pkl")

            save_solution(c, log_cb, f, output_path)
        print(f"* Meta Model generated and saved at: pretrained/{data_path.name}_{opts.beta}_{opts.theta}_{consts.WARM_START[opts.warm_start]}.joblib")

    elif opts.mode == consts.BRANCHING:
        # Solve multiple instances in parallel using SLURM array jobs
        if opts.strategy > 2:
            assert opts.theta2 > 0, "Theta must be set"
        if opts.warm_start != 0:
            assert opts.beta > 0 and opts.theta > 0 and opts.theta2 > 0, "Beta, theta, theta2 must be set"

        assert 0 <= opts.strategy <= len(consts.STRATEGY), "Unknown branching strategy"
        if opts.inst_parallel == 1:
            # total number of slurm workers detected
            # defaults to 1 if not running under SLURM
            N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

            # this worker's array index. Assumes slurm array job is zero-indexed
            # defaults to zero if not running under SLURM
            this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
            SCOREFILE = os.path.expanduser(
                f"./{data_path.name}_{opts.beta}_{opts.theta}_{opts.theta2}_{consts.WARM_START[opts.warm_start]}.csv"
            )
            for param_idx in range(this_worker, opts.dataset_size, N_WORKERS):
                instance_name = f"setcover_84_{data_path.name}_0.05_{param_idx}.lp"
                instance_path = data_path / instance_name

                c, log_cb, trained_model = solve_branching(instance_path, output_dir_path, opts)
                if c is None:
                    num_nodes = -1
                    total_time = -1
                else:
                    solve_status_id = c.solution.get_status()
                    if solve_status_id == c.solution.status.MIP_optimal:
                        num_nodes = c.solution.progress.get_num_nodes_processed()
                        total_time = log_cb.total_time
                    else:
                        num_nodes = -1
                        total_time = -1
                results = (param_idx, num_nodes, total_time)
                with open(SCOREFILE, "a") as f:
                    f.write(f'{",".join(map(str, results))}\n')
        else:
            # Solve single instance
            solve_branching(opts.instance, output_dir_path, opts)


def solve_optimal(instance_path, output_dir_path, opts):
    # Load instance
    # /scratch/rahulpat/setcover/train/1000_1000/1000_1000_0.lp
    instance_path = Path(instance_path)
    assert instance_path.exists(), "Instance not found!"


    valid_extensions = ['.lp', '.mps', '.mps.gz']
    assert instance_path.suffix in valid_extensions, "Invalid instance file format!"

    # Solve instance
    import cplex as CPX
    c = CPX.Cplex(str(instance_path))
    c.parameters.timelimit.set(opts.timelimit)
    disable_output(c)
    c.solve()

    # Fetch optimal objective value
    opt_dict = {}
    objective_value = None
    if c.solution.get_status() == c.solution.status.MIP_optimal:
        objective_value = c.solution.get_objective_value()
    opt_dict[instance_path.name] = objective_value

    # Save result
    output_dir_path = output_dir_path.joinpath("optimal_obj")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path.joinpath(instance_path.stem + ".pkl")
    pkl.dump(opt_dict, open(output_path, 'wb'))

def solve_branching(instance_path, output_dir_path, opts):
    # Load instance
    # /scratch/rahulpat/setcover/train/1000_1000/1000_1000_0.lp
    instance_path = Path(instance_path)
    assert instance_path.exists(), "Instance not found!"

    print(f'* Branching strategy: {consts.STRATEGY[opts.strategy]}')
    print(f"* File: {str(instance_path)}\n* Seed: {opts.seed}")

    # Load relevant solve_instance()
    baseline_strategy = (
            opts.strategy == consts.BS_DEFAULT
            or opts.strategy == consts.BS_SB
            or opts.strategy == consts.BS_PC
            or opts.strategy == consts.BS_SB_PC
    )
    if baseline_strategy:
        from strategy import baseline_solve_instance
        solve_instance = baseline_solve_instance
    else:
        from strategy import online_solve_instance
        solve_instance = online_solve_instance

    results = []

    opts_dict, primal_bound = get_opt_dict(output_dir_path, instance_path)

    print("* Loading Meta-model")
    meta_model = joblib.load(f'pretrained/{instance_path.parent.name}_{opts.beta}_{opts.theta}_{consts.WARM_START[opts.warm_start]}.joblib') if opts.warm_start != consts.NONE else None

    # /scratch/rahulpat/setcover/output/1000_1000/SB_PC
    beta_theta_dir = f"{opts.beta}_{opts.theta}_{opts.theta2}_{consts.WARM_START[opts.warm_start]}"
    output_dir_path = output_dir_path / consts.STRATEGY[opts.strategy] / beta_theta_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    # /scratch/rahulpat/setcover/output/1000_1000/SB_PC/1000_1000_0.pkl
    output_path = output_dir_path.joinpath(str(instance_path.stem) + ".pkl")
    

    if os.path.exists(output_path):
        print("* Solution already computed during meta-model training, aborting....")
        return None, None, None

    print("* Starting the solve...")
    c, log_cb, trained_model = solve_instance(
        path=str(instance_path),
        primal_bound=primal_bound,
        timelimit=opts.timelimit,
        branch_strategy=opts.strategy,
        seed=opts.seed,
        test=False,
        warm_start_model=meta_model,
        theta=opts.theta2,
    )


    save_solution(c, log_cb, instance_path, output_path)
    print(f"* Output file path: {output_path}")

    return c, log_cb, trained_model
if __name__ == "__main__":
    """
    * Usage instructions
    ----------------------------------------------------------------------------------------
    1. For generating optimal solutions
    python run.py --mode 0 --instance <instance path> --seed <seed>
    
    2. For branching
    python run.py --mode 1 --instance <instance path> --strategy <strategy_id> --seed <seed>
    
    * Parameters details
    ----------------------------------------------------------------------------------------     
    1. <instance_path> must be in the following format
        <some_directory>/<problem_type>/<problem_size>/<instance_name>
            <some_directory>    /home/rahul
            <problem_type>      /setcover
            <problem_size>      /1000_1000
            <instance_name>     /1000_1000_0.lp
            
    2. <strategy_id> can be between 0 to 5, where 
        0 ==> DEFAULT
        1 ==> Strong branching
        2 ==> Pseudocost branching
        3 ==> Strong(theta) + Pseudocost branching
        4 ==> Strong(theta) + SVM Rank
        5 ==> Strong(theta) + Feed forward Neural Network
        

    """
    opts = get_options(sys.argv[1:])
    run(opts)
