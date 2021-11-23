import cplex as CPX
import cplex.callbacks as CPX_CB

import consts
import params
from utils import get_logging_callback, set_params


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def __call__(self):
        self.times_called += 1

        # if self.times_called == 1:
        #     self.abort()


def solve_instance(path='set_cover.lp',
                   primal_bound=None,
                   timelimit=None,
                   seed=None,
                   test=True,
                   branch_strategy=consts.BS_SB_PC,
                   theta=params.THETA):
    # Read instance and set default parameters
    c = CPX.Cplex(path)
    set_params(c, primal_bound=primal_bound, timelimit=timelimit,
               seed=seed, test=test)

    log_cb = get_logging_callback(c)

    vsel_cb = c.register_callback(VariableSelectionCallback)
    vsel_cb.c = c
    vsel_cb.strategy = branch_strategy
    vsel_cb.times_called = 0
    vsel_cb.THETA = theta

    # Solve the instance and save stats
    c.solve()

    return c, log_cb
