import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np

import consts
import params
from utils import create_default_branches
from utils import get_candidates
from utils import get_data
from utils import get_logging_callback
from utils import get_sb_scores
from utils import set_params


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def __call__(self):
        # Check if the strategy is driven by CPLEX.
        # If yes, then create default branches
        if (self.branch_strategy == consts.CPX_DEFAULT
                or self.branch_strategy == consts.CPX_PC
                or self.branch_strategy == consts.CPX_SB):
            create_default_branches(self)

        # Find candidates for branching
        candidate_idxs = get_candidates(self)
        if len(candidate_idxs) == 0:
            return

        # Select branching variable idx
        self.times_called += 1
        branching_var_idx = None
        # Strong branching
        if self.branch_strategy == consts.BS_SB or (
                self.branch_strategy == consts.BS_SB_PC and self.times_called <= params.THETA):
            sb_scores, _ = get_sb_scores(self, candidate_idxs)
            if len(sb_scores):
                sb_scores = np.asarray(sb_scores)
                branching_var_idx = candidate_idxs[np.argmax(sb_scores)]
        # Pseudocode branching
        elif self.branch_strategy == consts.BS_PC or (
                self.branch_strategy == consts.BS_SB_PC and self.times_called > params.THETA):
            branching_var_idx = candidate_idxs[0]
        # Nothing to branch on
        if branching_var_idx is None:
            return

        branching_val = self.get_values(branching_var_idx)
        obj_val = self.get_objective_value()
        node_data = get_data(self)

        ##################################################################################
        # NOTE: branching_var must be an index of the variable
        branches = [(branching_var_idx, consts.LOWER_BOUND, np.floor(branching_val) + 1),
                    (branching_var_idx, consts.UPPER_BOUND, np.floor(branching_val))]
        ##################################################################################
        for branch in branches:
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            node_data_clone['branch_history'].append(branch)

            self.make_branch(obj_val, variables=[branch], constraints=[], node_data=node_data_clone)


def solve_instance(
        path='set_cover.lp',
        cutoff=None,
        timelimit=None,
        seed=None,
        test=True,
        branch_strategy=consts.BS_PC,
        theta=params.THETA,
        max_iterations=50,
        warm_start_model=None):
    # Read instance and set default parameters
    c = CPX.Cplex(path)
    np.random.seed(seed)
    set_params(c, cutoff=cutoff, timelimit=timelimit,
               branch_strategy=branch_strategy, seed=seed, test=test)

    log_cb = get_logging_callback(c)

    num_vars = c.variables.get_num()
    vsel_cb = c.register_callback(VariableSelectionCallback)
    vsel_cb.c = c
    vsel_cb.ordered_var_idx_lst = list(range(num_vars))
    
    vsel_cb.branch_strategy = branch_strategy
    vsel_cb.times_called = 0
    vsel_cb.THETA = theta
    vsel_cb.max_iterations = max_iterations
    vsel_cb.model = None

    # Solve the instance and save stats
    c.solve()
    return c, log_cb, vsel_cb
