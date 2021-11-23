import consts
import params
import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np


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
    
    # Single threaded
    c.parameters.threads.set(1)

    # Disable primal heuristics
    c.parameters.mip.strategy.heuristiceffort.set(0)

    # Diable presolve
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
        
    # Select from one of the default branching strategies
    if branch_strategy == consts.BS_DEFAULT:            
        c.parameters.mip.strategy.variableselect.set(0)        
    # elif branch_strategy == consts.BS_SB:
    #     c.parameters.mip.strategy.variableselect.set(3)
    # elif branch_strategy == consts.BS_PC:
    #     c.parameters.mip.strategy.variableselect.set(2)    
                
                        
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






def get_candidates(vsel_cb):
    """Get branching candidates based on Pseudocosts score"""
    pseudocosts = vsel_cb.get_pseudo_costs()
    values = vsel_cb.get_values()
    
    up_frac = np.ceil(values) - values
    down_frac = values - np.floor(values)
        
    scores = [(d * u * p[0] * p[1]) for p, d, u in zip(pseudocosts, down_frac, up_frac)]        
    
    variables = sorted(range(len(scores)), key=lambda i: -scores[i])

    candidates = []
    for i in variables:
        if len(candidates) == vsel_cb.K:
            break

        value = values[i]
        if abs(value-round(value)) > vsel_cb.EPSILON:
            candidates.append(i)

    return candidates


# def get_strong_branching_score(vsel_cb, candidates):
#     cclone = vsel_cb.get_clone()
#     status, parent_objval, dual_values = solve_as_lp(cclone)
#
#     vsel_cb.curr_node_dual_values = np.array(dual_values)
#     sb_scores = []
#     for cand in candidates:
#         status_lower, lower_objective = get_branch_solution(vsel_cb, cclone, cand, vsel_cb.LOWER_BOUND)
#         status_upper, upper_objective = get_branch_solution(vsel_cb, cclone, cand, vsel_cb.UPPER_BOUND)
#
#         delta_lower = max(lower_objective - parent_objval, vsel_cb.EPSILON)
#         delta_upper = max(upper_objective - parent_objval, vsel_cb.EPSILON)
#
#         sb_score = delta_lower * delta_upper
#         sb_scores.append(sb_score)
#
#         if status_lower != consts.OPTIMAL:
#             vsel_cb.num_infeasible_left[cand] += 1
#         if status_upper != consts.OPTIMAL:
#             vsel_cb.num_infeasible_right[cand] += 1
#
#     return np.asarray(sb_scores), cclone


def create_default_branches(node):
    for branch in range(node.get_num_branches()):
        node.make_cplex_branch(branch)


def get_child_obj(node, cclone, var, value, bound_type):
    get_bounds = None
    set_bounds = None
    new_bound = None
    if bound_type == consts.LOWER_BOUND:
        get_bounds = node.get_lower_bounds
        set_bounds = cclone.variables.set_lower_bounds
        new_bound = np.floor(value) + 1
    elif bound_type == consts.UPPER_BOUND:
        get_bounds = node.get_upper_bounds
        set_bounds = cclone.variables.set_upper_bounds
        new_bound = np.floor(value)

    # Update original bound and solve LP
    original_bound = get_bounds(var)
    set_bounds(var, new_bound)
    status, objective, _ = solve_as_lp(cclone, max_iterations=50)

    # Reset original bound
    set_bounds(var, original_bound)

    return status, objective


def solve_as_lp(c, max_iterations=50):
    dual_values = None
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


def apply_branch_history(c, branch_history):
    """Set the bound of variables in model c according to the
    provided branch_history
    """
    for b in branch_history:
        b_var = b[0]
        b_type = b[1]
        b_val = b[2]

        if b_type == consts.LOWER_BOUND:
            c.variables.set_lower_bounds(b_var, b_val)
        elif b_type == consts.UPPER_BOUND:
            c.variables.set_upper_bounds(b_var, b_val)


def get_data(node):
    # print('******')
    node_data = node.get_node_data()
    # print('+++++')
    if node_data is None:
        node_data = {'branch_history': []}

    return node_data


def get_clone(node):
    """Make a clone of the current node in the B&B tree"""

    # Make a clone of the original problem
    cclone = CPX.Cplex(node.c)
    # Get the branching history of the current node
    node_data = get_data(node)
    # Apply the branching history on the cloned node
    apply_branch_history(cclone, node_data['branch_history'])

    return cclone


def initialize_pseudocosts(node):
    # Solve root LP
    cclone = get_clone(node)
    status, parent_obj, dual_values = solve_as_lp(cclone)
    # print(parent_obj)
    if status == consts.LP_INFEASIBLE:
        print("\nRoot LP infeasible. Aborting...\n")
        node.abort()

    node.curr_node_dual_values = np.array(dual_values)
    lp_soln = cclone.solution.get_values(node.var_idx_lst)
    # print(lp_soln[:5])
    # Pseudocosts with strong branching initialization
    for var in node.var_idx_lst:
        value = lp_soln[var]
        # value = cclone.solution.get_values(var)
        down_frac = value - np.floor(value)
        up_frac = np.ceil(value) - value

        if np.abs(value - np.round(value)) <= params.EPSILON:
            continue

        up_status, up_obj = get_child_obj(node, cclone, var, value, consts.LOWER_BOUND)
        down_status, down_obj = get_child_obj(node, cclone, var, value, consts.UPPER_BOUND)

        # Update up pseudocost
        if up_status == consts.LP_OPTIMAL or up_status == consts.LP_ABORT_IT_LIM:
            up_delta = up_obj - parent_obj
            up_delta /= up_frac
            node.up_delta_sum[var] += up_delta
            node.up_count[var] += 1
            node.up_pc[var] = up_delta

        # Update down pseudocost
        if down_status == consts.LP_OPTIMAL or down_status == consts.LP_ABORT_IT_LIM:
            down_delta = down_obj - parent_obj
            down_delta /= down_frac
            node.down_delta_sum[var] += down_delta
            node.down_count[var] += 1
            node.down_pc[var] = down_delta

    # Set delta for infeasible up and down children to the average up and down delta sum
    up_pc_avg = [node.up_pc[k] for k, v in node.up_count.items() if v > 0]
    up_pc_avg = np.mean(up_pc_avg)
    for k, v in node.up_count.items():
        if v == 0:
            node.up_pc[k] = up_pc_avg
            # node.up_delta_sum += up_delta_sum_avg
            # node.up_count[k] += 1

    down_pc_avg = [node.down_pc[k] for k, v in node.down_count.items() if v > 0]
    down_pc_avg = np.mean(down_pc_avg)
    for k, v in node.down_count.items():
        if v == 0:
            node.down_pc[k] = down_pc_avg
            # node.down_delta_sum += down_pc_avg
            # node.down_count[k] += 1

