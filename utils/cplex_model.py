import pathlib
import pickle as pkl
import time
from operator import itemgetter

import cplex as CPX
import cplex.callbacks as CPX_CB
import joblib
import numpy as np

import consts
import params
from models.MLPClassifier import MLPClassifier1 as MLPClassifier


class LoggingCallback(CPX_CB.MIPInfoCallback):
    def __call__(self):
        self.total_time = self.get_time() - self.get_start_time()


def get_logging_callback(c):
    log_cb = c.register_callback(LoggingCallback)
    log_cb.total_time = 0

    return log_cb


def set_params(c, primal_bound=None,
               branch_strategy=None,
               timelimit=None,
               seed=None,
               test=False):
    if seed is not None:
        c.parameters.randomseed.set(seed)

    # Select from one of the default branching strategies
    if branch_strategy == consts.CPX_DEFAULT:
        c.parameters.mip.strategy.variableselect.set(0)
    elif branch_strategy == consts.CPX_PC:
        c.parameters.mip.strategy.variableselect.set(2)
    elif branch_strategy == consts.CPX_SB:
        c.parameters.mip.strategy.variableselect.set(3)

    # Single threaded
    c.parameters.threads.set(1)

    # Disable primal heuristics
    c.parameters.mip.strategy.heuristiceffort.set(0)

    # Disable presolve
    c.parameters.preprocessing.presolve.set(0)

    c.parameters.mip.tolerances.integrality.set(params.EPSILON)

    c.parameters.mip.strategy.search.set(
        c.parameters.mip.strategy.search.values.traditional)

    # Set the primal bound if provided
    if primal_bound is not None:
        if c.objective.get_sense() == consts.MINIMIZE:
            c.parameters.mip.tolerances.lowercutoff.set(primal_bound)
        else:
            c.parameters.mip.tolerances.uppercutoff.set(primal_bound)

    # Set timelimit if provided
    if timelimit is not None:
        c.parameters.timelimit.set(timelimit)

    # Disable cutting planes
    c.parameters.mip.limits.eachcutlimit.set(0)
    c.parameters.mip.cuts.bqp.set(-1)
    c.parameters.mip.cuts.cliques.set(-1)
    c.parameters.mip.cuts.covers.set(-1)
    c.parameters.mip.cuts.disjunctive.set(-1)
    c.parameters.mip.cuts.flowcovers.set(-1)
    c.parameters.mip.cuts.pathcut.set(-1)
    c.parameters.mip.cuts.gomory.set(-1)
    c.parameters.mip.cuts.gubcovers.set(-1)
    c.parameters.mip.cuts.implied.set(-1)
    c.parameters.mip.cuts.localimplied.set(-1)
    c.parameters.mip.cuts.liftproj.set(-1)
    c.parameters.mip.cuts.mircut.set(-1)
    c.parameters.mip.cuts.mcfcut.set(-1)
    c.parameters.mip.cuts.rlt.set(-1)
    c.parameters.mip.cuts.zerohalfcut.set(-1)

    if not test:
        disable_output(c)


def disable_output(c):
    c.set_log_stream(None)
    c.set_error_stream(None)
    c.set_warning_stream(None)
    c.set_results_stream(None)


def create_default_branches(context):
    for branch in range(context.get_num_branches()):
        context.make_cplex_branch(branch)


def get_optimal_obj_dict(output_path, instance_path):
    # Check if optimal solution exists to provide as primal bound
    primal_bound = 1e6
    opt_dict = None
    optimal_obj_path = output_path.joinpath(f"optimal_obj/{instance_path.stem}.pkl")
    print(f"* Checking optimal objective pickle at {optimal_obj_path}...")
    if optimal_obj_path.exists():
        opt_dict = pkl.load(open(optimal_obj_path, 'rb'))
        key = str(instance_path)
        if key in opt_dict and opt_dict[key]['status'] == consts.MIP_OPTIMAL:
            primal_bound = opt_dict[key]['objective_value']
            print(f"\t** Primal bound: {primal_bound}")
        else:
            print(f"\t** Instance primal bound not found...")
    else:
        print("\t** Warning: Optimal objective pickle not found. Can't set primal bound.")

    return opt_dict, primal_bound


def solve_as_lp(c, max_iterations=50):
    disable_output(c)
    # Create LP for the input MIP
    c.set_problem_type(c.problem_type.LP)
    # Set the maximum number of iterations for solving the LP
    if max_iterations is not None:
        c.parameters.simplex.limits.iterations = max_iterations

    c.solve()
    status, objective, dual_values = None, None, None

    status = c.solution.get_status()
    if status == consts.LP_OPTIMAL or status == consts.LP_ABORT_IT_LIM:
        objective = c.solution.get_objective_value()
        dual_values = c.solution.get_dual_values()

    return status, objective, dual_values


def get_branch_solution(context, cclone, var_idx, bound_type):
    value = context.get_values(var_idx)

    get_bounds = None
    set_bounds = None
    new_bound = None
    if bound_type == consts.LOWER_BOUND:
        get_bounds = context.get_lower_bounds
        set_bounds = cclone.variables.set_lower_bounds
        new_bound = np.floor(value) + 1
    elif bound_type == consts.UPPER_BOUND:
        get_bounds = context.get_upper_bounds
        set_bounds = cclone.variables.set_upper_bounds
        new_bound = np.floor(value)

    original_bound = get_bounds(var_idx)

    set_bounds(var_idx, new_bound)
    status, objective, _ = solve_as_lp(cclone)
    set_bounds(var_idx, original_bound)

    return status, objective


def apply_branch_history(c, branch_history):
    for b in branch_history:
        b_var_idx = b[0]
        b_type = b[1]
        b_val = b[2]

        if b_type == consts.LOWER_BOUND:
            c.variables.set_lower_bounds(b_var_idx, b_val)
        elif b_type == consts.UPPER_BOUND:
            c.variables.set_upper_bounds(b_var_idx, b_val)


def get_data(context):
    node_data = context.get_node_data()
    if node_data is None:
        node_data = {'branch_history': []}

    return node_data


def get_clone(context):
    cclone = CPX.Cplex(context.c)

    node_data = get_data(context)
    apply_branch_history(cclone, node_data['branch_history'])

    return cclone


# def get_candidates(pseudocosts, values, branch_strategy, var_idx_lst):
#     up_frac = np.ceil(values) - values
#     down_frac = values - np.floor(values)
#
#     scores = [(uf * df * pc[0] * pc[1], vidx)
#               for pc, uf, df, vidx in zip(pseudocosts, up_frac, down_frac, var_idx_lst)]
#     scores = sorted(scores, key=itemgetter(0), reverse=True)
#
#     num_candidates = params.K if branch_strategy != consts.BS_PC else 1
#     candidate_idxs = []
#     ranked_var_idx_lst = [i[1] for i in scores]
#     for var_idx in ranked_var_idx_lst:
#         if len(candidate_idxs) == num_candidates:
#             break
#
#         value = values[var_idx]
#         if not abs(value - round(value)) <= params.EPSILON:
#             candidate_idxs.append(var_idx)
#
#     return candidate_idxs


def get_candidates(pseudocosts, values, values_dict, branch_strategy, var_name_lst):
    up_frac = np.ceil(values) - values
    down_frac = values - np.floor(values)

    scores = [(uf * df * pc[0] * pc[1], vname)
              for pc, uf, df, vname in zip(pseudocosts, up_frac, down_frac, var_name_lst)]
    scores = sorted(scores, key=itemgetter(0), reverse=True)
    # print("-----------", scores[:20])
    ranked_var_name_lst = [i[1] for i in scores]

    num_candidates = params.K if branch_strategy != consts.BS_PC else 1
    candidate_names = []
    for name in ranked_var_name_lst:
        if len(candidate_names) == num_candidates:
            break

        value = values_dict[name]
        if not abs(value - round(value)) <= params.EPSILON:
            candidate_names.append(name)

    return candidate_names


def get_sb_scores(context, candidate_idxs):
    cclone = get_clone(context)
    status, parent_objective, dual_values = solve_as_lp(cclone)

    sb_scores = []
    if status == consts.LP_OPTIMAL or status == consts.LP_ABORT_IT_LIM:
        context.curr_node_dual_values = np.asarray(dual_values)
        for var_idx in candidate_idxs:
            upper_status, upper_objective = get_branch_solution(context, cclone, var_idx, consts.LOWER_BOUND)
            lower_status, lower_objective = get_branch_solution(context, cclone, var_idx, consts.UPPER_BOUND)

            # Infeasibility leads to higher score as it helps in pruning the tree
            if upper_status == consts.LP_INFEASIBLE:
                upper_objective = consts.INFEASIBILITY_SCORE
                context.num_infeasible_right[var_idx] += 1
            if lower_status == consts.LP_INFEASIBLE:
                lower_objective = consts.INFEASIBILITY_SCORE
                context.num_infeasible_left[var_idx] += 1

            # Calculate deltas
            delta_upper = max(upper_objective - parent_objective, params.EPSILON)
            delta_lower = max(lower_objective - parent_objective, params.EPSILON)

            # Calculate sb score
            sb_score = delta_lower * delta_upper
            sb_scores.append(sb_score)

    else:
        print("Root LP infeasible...")

    return sb_scores, cclone


def save_mip_solve_info(c, instance_path, output_path, total_time, branch_calls=None):
    """ Save relevant information after a MIP solve like status, objective value,
    time, # branching calls, and # nodes.

    :param c: cplex.Cplex object
        The object on which solve was performed
    :param instance_path: pathlib.Path object
        Path of instance solved
    :param output_path: pathlib.Path object
        Path to solve the solve information
    :param total_time: float
        Wallclock time elapsed for solve
    :param branch_calls: int
        # of branching calls
    """
    # Get the # nodes, time elapsed during the solve and # branching calls
    num_nodes = c.solution.progress.get_num_nodes_processed()

    # Get solve status and objective
    objective_value = None
    solve_status_id = c.solution.get_status()
    solve_status_verbose = c.solution.status[c.solution.get_status()]
    if c.solution.is_primal_feasible():
        objective_value = c.solution.get_objective_value()

    print(f"\n  I: {instance_path.name} S: {solve_status_verbose}, T: {total_time}, BC: {branch_calls}, "
          f"N: {num_nodes}, OBJ: {objective_value}\n")

    # Prepare result
    result_dict = {str(instance_path): {'status': solve_status_id,
                                        'status_verbose': solve_status_verbose,
                                        'objective_value': objective_value,
                                        'total_time': total_time,
                                        'num_nodes': num_nodes,
                                        'branch_calls': branch_calls}}

    # Save results
    print(f'* Saving mip solve result to: {output_path}')
    pkl.dump(result_dict, open(output_path, 'wb'))
    # print(result_dict)


def find_cutoff(instance_path, output_path, opts):
    assert instance_path.exists(), "Instance not found!"

    valid_extensions = ['.lp', '.mps', '.mps.gz']
    assert instance_path.suffix in valid_extensions, "Invalid instance file format!"

    # Solve instance

    c = CPX.Cplex(str(instance_path))
    c.parameters.timelimit.set(opts.timelimit)
    c.parameters.threads.set(4)
    disable_output(c)
    tick = time.time()
    c.solve()
    total_time = time.time() - tick

    # Save result
    _output_path = output_path.joinpath("optimal_obj")
    _output_path.mkdir(parents=True, exist_ok=True)
    _output_path = _output_path.joinpath(instance_path.stem + ".pkl")
    save_mip_solve_info(c, instance_path, _output_path, total_time)


def update_meta_model_param(meta_model_param, new_model, iter, opts):
    warm_start_model = None
    if iter < opts.beta:
        # Incremental Averaging of weights
        if opts.warm_start == consts.AVERAGE_MODEL:
            if meta_model_param is not None:
                for i in range(len(new_model.coefs_)):
                    meta_model_param[0][i] += meta_model_param[0][i] + (
                            new_model.coefs_[i] - meta_model_param[0][i]) / (iter + 1)
                    meta_model_param[1][i] += meta_model_param[1][i] + (
                            new_model.intercepts_[i] - meta_model_param[1][i]) / (iter + 1)
            else:
                meta_model_param = (new_model.coefs_, new_model.intercepts_)
        elif opts.warm_start == consts.INCREMENTAL_WARM_START:
            warm_start_model = new_model
            new_model.n_iter_no_change += 100

    if iter == opts.beta - 1:
        # Initialize meta model and save for future use
        if (opts.warm_start == consts.AVERAGE_MODEL and iter == opts.beta - 1):
            warm_start_model = MLPClassifier(init_params=meta_model_param, learning_rate_init=0.01,
                                             n_iter_no_change=100, max_iter=300, warm_start=True)
        dataset_type = pathlib.Path(opts.dataset).name
        joblib.dump(warm_start_model,
                    f'pretrained/{dataset_type}_{opts.beta}_{opts.theta}_{consts.WARM_START[opts.warm_start]}.joblib')

    return meta_model_param, warm_start_model


def solve_branching(instance_path, output_path, opts, theta, warm_start_model=None):
    # Load instance
    assert instance_path.exists(), "Instance not found!"

    print(f'* Branching strategy: {consts.STRATEGY[opts.strategy]}')
    print(f"* File: {str(instance_path)}\n* Seed: {opts.seed}")

    opts_dict, primal_bound = get_optimal_obj_dict(output_path, instance_path)
    if opts_dict is None or primal_bound is None or primal_bound == 1e6:
        return None, None, None
    if opts.mode != consts.TRAIN_META:
        print("* Loading Meta-model")
        meta_model = joblib.load(
            f'pretrained/{instance_path.parent.name}_{opts.beta}_{opts.theta}'
            f'_{consts.WARM_START[opts.warm_start]}.joblib') \
            if opts.warm_start != consts.NONE else None
    else:
        meta_model = warm_start_model
    if meta_model is None:
        print('\t** No meta-model found!')

    beta_theta_dir = f"{opts.beta}_{opts.theta}_{opts.theta2}_{consts.WARM_START[opts.warm_start]}"
    output_path1 = output_path / consts.STRATEGY[opts.strategy] / beta_theta_dir
    output_path1.mkdir(parents=True, exist_ok=True)
    output_path1 = output_path1.joinpath(str(instance_path.stem) + ".pkl")

    if output_path1.exists():
        print("* Solution already computed during meta-model training, aborting....")
        return None, None, None

    print("* Starting the solve...")
    # Load relevant solve_instance()
    baseline_strategy = (
            opts.strategy == consts.CPX_DEFAULT
            or opts.strategy == consts.CPX_PC
            or opts.strategy == consts.CPX_SB
            or opts.strategy == consts.BS_SB
            or opts.strategy == consts.BS_PC
            or opts.strategy == consts.BS_SB_PC
    )
    if baseline_strategy:
        from strategy import baseline_solve_instance as solve_instance
    else:
        from strategy import online_solve_instance as solve_instance
    c, log_cb, vsel_cb = solve_instance(
        path=str(instance_path),
        primal_bound=primal_bound,
        timelimit=opts.timelimit,
        branch_strategy=opts.strategy,
        seed=opts.seed,
        test=False,
        warm_start_model=meta_model,
        theta=theta,
    )

    save_mip_solve_info(c, instance_path, output_path1, log_cb.total_time,
                        branch_calls=vsel_cb.times_called)

    return c, log_cb, vsel_cb
