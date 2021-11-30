import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np

import consts
import params
from utils import get_candidates
from utils import get_data
from utils import get_logging_callback
from utils import get_sb_scores
from utils import set_params


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def __call__(self):

        if (self.branch_strategy == consts.CPX_DEFAULT
                or self.branch_strategy == consts.CPX_PC
                or self.branch_strategy == consts.CPX_SB):
            return
        # Find candidates for branching
        # pseudocosts = self.get_pseudo_costs(self.var_idx_lst)
        pseudocosts = self.get_pseudo_costs(self.var_name_lst)
        pseudocosts_dict = {k: v for k, v in zip(self.var_name_lst, pseudocosts)}

        # values = self.get_values(self.var_idx_lst)
        values = self.get_values(self.var_name_lst)
        values_dict = {k: v for k, v in zip(self.var_name_lst, values)}

        # candidate_idxs = get_candidates(pseudocosts, values, self.branch_strategy, self.var_idx_lst)
        candidate_names = get_candidates(pseudocosts, values, values_dict, self.branch_strategy, self.var_name_lst)
        # print("++++++++++", candidate_names)

        # if len(candidate_idxs) == 0:
        #     return
        if len(candidate_names) == 0:
            return
        else:
            candidate_idxs = [self.var_name_idx_dict[k] for k in candidate_names]

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
        # print("*********", branching_var_idx, pseudocosts_dict['x394'], pseudocosts_dict['x395'],
            #   pseudocosts_dict['x393'])

        # print("\t********", branching_var_idx, pseudocosts[394])
        branching_val = self.get_values(branching_var_idx)
        obj_val = self.get_objective_value()

        node_data = get_data(self)

        # NOTE: branching_var must be an index of the variable
        branches = [(branching_var_idx, consts.LOWER_BOUND, np.floor(branching_val) + 1),
                    (branching_var_idx, consts.UPPER_BOUND, np.floor(branching_val))]

        for branch in branches:
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            node_data_clone['branch_history'].append(branch)

            self.make_branch(obj_val, variables=[branch], constraints=[], node_data=node_data_clone)

        # if self.times_called == 5:
        #     self.abort()


def solve_instance(
        path='set_cover.lp',
        primal_bound=None,
        timelimit=None,
        seed=None,
        test=True,
        branch_strategy=consts.BS_PC,
        theta=params.THETA,
        warm_start_model=None):
    # Read instance and set default parameters
    c = CPX.Cplex(path)
    np.random.seed(seed)
    set_params(c, primal_bound=primal_bound, timelimit=timelimit,
               branch_strategy=branch_strategy, seed=seed, test=test)

    log_cb = get_logging_callback(c)

    vsel_cb = c.register_callback(VariableSelectionCallback)
    vsel_cb.c = c
    var_name_lst = c.variables.get_names()
    vsel_cb.var_name_lst = var_name_lst
    var_idx_lst = c.variables.get_indices(var_name_lst)
    vsel_cb.var_idx_lst = var_idx_lst
    vsel_cb.var_name_idx_dict = {i[0]: i[1] for i in zip(var_name_lst, var_idx_lst)}

    # for i in zip(var_name_lst, var_idx_lst):
    #     print(i[0], i[1])

    vsel_cb.branch_strategy = branch_strategy
    vsel_cb.times_called = 0
    vsel_cb.THETA = theta
    vsel_cb.model = None

    # Solve the instance and save stats
    c.solve()
    return c, log_cb, vsel_cb
