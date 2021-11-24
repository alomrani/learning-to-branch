import multiprocessing as mp
import pickle as pkl
import sys
from pathlib import Path

import consts
from options import get_options
from utils import disable_output


def worker(solve_instance, f, primal_bound, branch_strategy, timelimit, seed):
    print(f"    {str(f)}, {seed}")
    c, log_cb = solve_instance(path=str(f),
                               primal_bound=primal_bound,
                               timelimit=timelimit,
                               branch_strategy=branch_strategy,
                               seed=seed,
                               test=False)

    solve_status_id = c.solution.get_status()
    solve_status_verbose = c.solution.status[c.solution.get_status()]

    key = f.stem + f'_{seed}.lp'
    result_dict = {key: {'status': solve_status_id,
                         'status_verbose': solve_status_verbose}}
    if solve_status_id == c.solution.status.MIP_optimal:
        result_dict[key]['total_time'] = log_cb.total_time
        result_dict[key]['num_nodes'] = log_cb.num_nodes
        print(f"\tS: {solve_status_verbose}, T: {log_cb.total_time}, N:{log_cb.num_nodes}\n")
    else:
        print(f"\tS: {solve_status_verbose}, T: {log_cb.total_time}, N:{log_cb.num_nodes}\n")

    return result_dict


def run(opts):
    print(f'* Run mode: {consts.MODE[opts.mode]}')
    if opts.mode == consts.GENERATE_OPTIMAL:
        import cplex as CPX

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
        import params

        assert 0 <= opts.strategy <= len(consts.STRATEGY), "Unknown branching strategy"
        print(f'* Branching strategy: {consts.STRATEGY[opts.strategy]}')

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

        for f in train_path.glob('*.lp'):
            # Only process instances that are solved by the CPLEX to
            # optimality and use their optimal objective value as primal bound
            if f.name in opt_dict.keys():
                primal_bound = opt_dict[f.name]
                for seed in params.SEEDS:
                    if opts.inst_parallel:
                        results.append(pool.apply_async(worker,
                                                        args=(solve_instance, f,
                                                              primal_bound,
                                                              opts.strategy,
                                                              opts.timelimit,
                                                              seed,)))
                    else:
                        results.append(worker(solve_instance, f, primal_bound, opts.strategy,
                                              opts.timelimit, seed))
                    break
            break

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
