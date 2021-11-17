import consts
import cplex.callbacks as CPX_CB


class LoggingCallback(CPX_CB.MIPInfoCallback):
    def __call__(self):
        nn = self.get_num_nodes()
        if nn > self.num_nodes:
            self.num_nodes = nn

        self.total_time = self.get_time() - self.get_start_time()

        
def set_params(c, primal_bound, test=False):
    # Single threaded
    c.parameters.threads.set(1)

    # Disable primal heuristics
    c.parameters.mip.strategy.heuristiceffort.set(0)

    # Set the primal bound if provided    
    if primal_bound is not None:
        if c.objective.get_sense() == MINIMIZE:
            c.parameters.mip.tolerances.lowercutoff.set(primal_bound)
        else:
            c.parameters.mip.tolerances.uppercutoff.set(primal_bound)

    if not test:
        disable_output(c)


def disable_cuts(c):
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


def disable_output(c):
    c.set_log_stream(None)
    c.set_error_stream(None)
    c.set_warning_stream(None)
    c.set_results_stream(None)


def solve_as_lp(c, max_iterations=None):    
    dual_values = None
    disable_output(c)
    c.set_problem_type(c.problem_type.LP)
    # Set the maximum number of iterations for solving the LP
    if max_iterations is not None:
        c.parameters.simplex.limits.iterations = max_iterations

    c.solve()
    status = c.solution.get_status()    
    objective = c.solution.get_objective_value() if status == consts.OPTIMAL else consts.INFEASIBILITY
    # BUG: Access dual solution only if status is optimal or feasible
    dual_values = c.solution.get_dual_values()
    return status, objective, dual_values


def apply_branch_history(c, branch_history):
    for b in branch_history:
        b_var = b[0]
        b_type = b[1]
        b_val = b[2]

        if b_type == consts.LOWER_BOUND:
            c.variables.set_lower_bounds(b_var, b_val)
        elif b_type == consts.UPPER_BOUND:
            c.variables.set_upper_bounds(b_var, b_val)
            

def get_logging_callback(c):    
    log_cb = c.register_callback(LoggingCallback)
    log_cb.num_nodes = 0
    log_cb.total_time = 0
    
    return log_cb
