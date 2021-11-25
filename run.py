import multiprocessing as mp
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

def worker(solve_instance, f, primal_bound, branch_strategy, timelimit, seed, theta, warm_start_model=None):
    print(f"    {str(f)}, {seed}")

    c, log_cb, trained_model = solve_instance(
        warm_start_model=warm_start_model,
        path=str(f),
        primal_bound=primal_bound,
        timelimit=timelimit,
        theta=theta,
        branch_strategy=branch_strategy,
        seed=seed,
        test=False,
    )

    solve_status_id = c.solution.get_status()
    solve_status_verbose = c.solution.status[c.solution.get_status()]

    key = f.stem + f'_{seed}.lp'
    result_dict = {key: {'status': solve_status_id,
                         'status_verbose': solve_status_verbose},
                        'trained_model': trained_model}
    num_nodes = c.solution.progress.get_num_nodes_processed()
    if solve_status_id == c.solution.status.MIP_optimal:
        result_dict[key]['total_time'] = log_cb.total_time
        result_dict[key]['num_nodes'] = log_cb.num_nodes
        print(f"\tS: {solve_status_verbose}, T: {log_cb.total_time}, N:{num_nodes}\n")
    else:
        print(f"\tS: {solve_status_verbose}, T: {log_cb.total_time}, N:{num_nodes}\n")

    return result_dict


def update_meta_model_param(meta_model_param, new_model, iter, opts):

    warm_start_model = None
    if iter < opts.beta:
        # Incremental Averaging of weights
        if opts.warm_start == 2:
            if meta_model_param is not None:
                for i in range(len(new_model.coefs_)):
                    meta_model_param[0][i] += meta_model_param[0][i] + (new_model.coefs_[i] - meta_model_param[0][i]) / (iter + 1)
                    meta_model_param[1][i] += meta_model_param[1][i] + (new_model.intercepts_[i] - meta_model_param[1][i]) / (iter + 1)
            else:
                meta_model_param = (new_model.coefs_, new_model.intercepts_)

    if iter == opts.beta - 1:
        # Initialize meta model and save for future use
        if (opts.warm_start == 2 and iter == opts.beta - 1):
            warm_start_model = MLPClassifier(verbose=True, init_params=meta_model_param, learning_rate_init=0.01, n_iter_no_change=50, max_iter=300, warm_start=True)
        elif opts.warm_start == 3:
            warm_start_model = new_model
        joblib.dump(warm_start_model, f'pretrained/{opts.beta}_{opts.theta}.joblib')

    # Use already saved meta model
    if opts.warm_start == 1 or iter > opts.beta:
        warm_start_model = joblib.load(f'pretrained/{opts.beta}_{opts.theta}.joblib')
    

    return meta_model_param, warm_start_model


def run(opts):
    print(f'* Run mode: {consts.MODE[opts.mode]}')
    if opts.mode == consts.GENERATE_OPTIMAL:


        train_path = Path(opts.train_dataset)
        output_path = Path(opts.output_dir)
        output_path = output_path / "train"
        output_path.mkdir(parents=True, exist_ok=True)

        opt_dict = {}
        for f in train_path.glob('*.lp'):
            print("*", str(f))
            # print(f.name)
            c = CPX.Cplex(str(f))
            disable_output(c)
            c.solve()
            if c.solution.get_status() == c.solution.status.MIP_optimal:
                opt_dict[f.name] = c.solution.get_objective_value()

        pkl.dump(opt_dict, open(output_path / 'optimal_obj.pkl', 'wb'))

    elif opts.mode == consts.BRANCHING:

        assert 0 <= opts.strategy <= len(consts.STRATEGY), "Unknown branching strategy"
        print(f'* Branching strategy: {consts.STRATEGY[opts.strategy]}')
        print(f'* Warm-start strategy: {consts.WARM_START[opts.warm_start]}')
        baseline_strategy = (
                opts.strategy == consts.BS_DEFAULT
                or opts.strategy == consts.BS_SB
                or opts.strategy == consts.BS_PC
                or opts.strategy == consts.BS_SB_PC
        )

        solve_instance = None
        if baseline_strategy:
            from strategy import baseline_solve_instance
            solve_instance = baseline_solve_instance
        else:
            from strategy import online_solve_instance
            solve_instance = online_solve_instance

        train_path = Path(opts.train_dataset)
        output_path = Path(opts.output_dir)

        results = []
        opt_dict = pkl.load(open(output_path / 'train/optimal_obj.pkl', 'rb'))
        if opts.inst_parallel:
            pool = mp.Pool(processes=opts.num_workers)
        meta_model_param, warm_start_model = None, None
        theta = opts.theta
        for i, f in enumerate(train_path.glob('*.lp')):
            # Only process instances that are solved by the CPLEX to
            # optimality and use their optimal objective value as primal bound
            if f.name in opt_dict.keys():
                if i >= opts.beta:
                    theta = opts.theta2
                primal_bound = opt_dict[f.name]
                for seed in params.SEEDS:
                    if opts.inst_parallel:
                        results.append(pool.apply_async(worker,
                                                        args=(solve_instance, f,
                                                              primal_bound,
                                                              opts.strategy,
                                                              opts.timelimit,
                                                              seed, opts)))
                    else:
                        results.append(worker(solve_instance, f, primal_bound, opts.strategy,
                                              opts.timelimit, seed, warm_start_model=warm_start_model, theta=theta))

                meta_model_param, warm_start_model = update_meta_model_param(meta_model_param, results[-1]["trained_model"], i, opts)

        # Wait for the workers to get finish
        if opts.inst_parallel:
            results = [r.get() for r in results]

        # Prepare results dict
        results_dict = {}
        for d in results:
            results_dict.update(d)

        # Save results
        ofp = output_path / f"train/result_{consts.STRATEGY[opts.strategy]}.pkl"
        pkl.dump(results_dict, open(ofp, 'wb'))


if __name__ == "__main__":
    opts = get_options(sys.argv[1:])
    run(opts)
