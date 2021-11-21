import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np

import consts
import params


class LoggingCallback(CPX_CB.MIPInfoCallback):
    def __call__(self):
        nn = self.get_num_nodes()
        if nn > self.num_nodes:
            self.num_nodes = nn

        self.total_time = self.get_time() - self.get_start_time()


def get_logging_callback(c):
    log_cb = c.register_callback(LoggingCallback)
    log_cb.num_nodes = 0
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
    if branch_strategy == consts.BS_DEFAULT:
        c.parameters.mip.strategy.variableselect.set(0)

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


def solve_as_lp(c, max_iterations=None):
    disable_output(c)
    # Create LP for the input MIP
    c.set_problem_type(c.problem_type.LP)
    # Set the maximum number of iterations for solving the LP
    if max_iterations is not None:
        c.parameters.simplex.limits.iterations = max_iterations

    c.solve()
    status, objective, dual_values = None, 1e6, None

    status = c.solution.get_status()
    if status == consts.LP_OPTIMAL or status == consts.LP_ABORT_IT_LIM:
        objective = c.solution.get_objective_value()
        dual_values = c.solution.get_dual_values()

    return status, objective, dual_values


def get_branch_solution(context, cclone, cand, bound_type):
    cand_val = context.get_values(cand)

    get_bounds = None
    set_bounds = None
    new_bound = None
    if bound_type == consts.LOWER_BOUND:
        get_bounds = context.get_lower_bounds
        set_bounds = cclone.variables.set_lower_bounds
        new_bound = np.floor(cand_val) + 1
    elif bound_type == consts.UPPER_BOUND:
        get_bounds = context.get_upper_bounds
        set_bounds = cclone.variables.set_upper_bounds
        new_bound = np.floor(cand_val)

    original_bound = get_bounds(cand)

    set_bounds(cand, new_bound)
    status, objective, _ = solve_as_lp(cclone, max_iterations=50)
    set_bounds(cand, original_bound)

    return status, objective


def apply_branch_history(c, branch_history):
    for b in branch_history:
        b_var = b[0]
        b_type = b[1]
        b_val = b[2]

        if b_type == consts.LOWER_BOUND:
            c.variables.set_lower_bounds(b_var, b_val)
        elif b_type == consts.UPPER_BOUND:
            c.variables.set_upper_bounds(b_var, b_val)


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


def get_candidates(pseudocosts, values, branch_strategy):
    up_frac = np.ceil(values) - values
    down_frac = values - np.floor(values)

    scores = [uf * df * pc[0] * pc[1] for pc, uf, df in zip(pseudocosts,
                                                            up_frac,
                                                            down_frac)]
    variables = sorted(range(len(scores)), key=lambda i: -scores[i])

    num_candidates = params.K if branch_strategy != consts.BS_PC else 1
    candidates = []
    for i in variables:
        if len(candidates) == num_candidates:
            break

        value = values[i]
        if not abs(value - round(value)) <= params.EPSILON:
            candidates.append(i)

    return candidates
