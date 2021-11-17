import consts
import cplex as CPX
import cplex.callbacks as CPX_CB
import params
from consts import BS_DEFAULT, BS_PC, BS_SB, BS_SB_PC
from utils import disable_cuts, get_logging_callback, set_params


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def create_default_branches(self):
        for branch in range(self.get_num_branches()):
            self.make_cplex_branch(branch)

    def __call__(self):
        self.times_called += 1
        if self.times_called == 1:
            disable_cuts(self.c)

        # If strategy is SB_DEFAULT or PC_DEFAULT then shortcut
        if self.strategy == BS_DEFAULT:            
            self.c.parameters.mip.strategy.variableselect.set(0)
            self.create_default_branches()
            return

        elif self.strategy == BS_SB or (self.strategy == BS_SB_PC and self.times_called <= self.THETA):
            self.c.parameters.mip.strategy.variableselect.set(3)
            self.create_default_branches()
            return

        elif self.strategy == BS_PC or (self.strategy == BS_SB_PC and self.times_called > self.THETA):
            self.c.parameters.mip.strategy.variableselect.set(2)
            self.create_default_branches()
            return

        

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
