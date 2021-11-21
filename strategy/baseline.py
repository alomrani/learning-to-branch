import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np

import consts
import params
from utils import get_candidates
from utils import get_data
from utils import get_logging_callback
from utils import set_params


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def __call__(self):
        self.times_called += 1

        if self.branch_strategy == consts.BS_DEFAULT:
            return

        # Find candidates for branching
        pseudocosts = self.get_pseudo_costs()
        values = self.get_values()
        candidates = get_candidates(pseudocosts, values, self.branch_strategy)
        if len(candidates) == 0:
            return

        if self.branch_strategy == consts.BS_PC:
            branching_var = candidates[0]
            branching_val = self.get_values(branching_var)
            obj_val = self.get_objective_value()

            node_data = get_data(self)

            branches = [(branching_var, consts.LOWER_BOUND, np.floor(branching_val) + 1),
                        (branching_var, consts.UPPER_BOUND, np.floor(branching_val))]

            for branch in branches:
                node_data_clone = node_data.copy()
                node_data_clone['branch_history'] = node_data['branch_history'][:]
                node_data_clone['branch_history'].append(branch)

                self.make_branch(obj_val, variables=[branch], constraints=[], node_data=node_data_clone)

        # if self.times_called == 5:
        #     self.abort()


def solve_instance(path='set_cover.lp',
                   primal_bound=None,
                   timelimit=None,
                   seed=None,
                   test=True,
                   branch_strategy=consts.BS_PC,
                   theta=params.THETA):
    # Read instance and set default parameters
    c = CPX.Cplex(path)
    set_params(c, primal_bound=primal_bound, timelimit=timelimit,
               branch_strategy=branch_strategy, seed=seed, test=test)

    log_cb = get_logging_callback(c)

    vsel_cb = c.register_callback(VariableSelectionCallback)
    vsel_cb.c = c
    var_lst = c.variables.get_names()
    vsel_cb.var_lst = var_lst
    var_idx_lst = c.variables.get_indices(var_lst)
    vsel_cb.var_idx_lst = var_idx_lst

    vsel_cb.branch_strategy = branch_strategy
    vsel_cb.times_called = 0
    vsel_cb.THETA = theta

    # Solve the instance and save stats
    c.solve()
    print(vsel_cb.times_called)
    return c, log_cb
