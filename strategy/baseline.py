from operator import itemgetter

import consts
import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np
import params
from consts import BS_DEFAULT, BS_PC, BS_SB, BS_SB_PC
from utils import (get_logging_callback, set_params, initialize_pseudocosts,
                   get_candidates, get_data, get_clone, get_child_obj)


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def __call__(self):
        self.times_called += 1
        print(self.times_called)
        branching_var = None

        # Initialize pseudocosts using strong branching scores
        if self.times_called == 1:
            initialize_pseudocosts(self)

        if branching_var is None:
            # Select candidates based on pseudocosts
            lp_soln = self.get_values()
            up_frac = np.ceil(lp_soln) - lp_soln
            down_frac = lp_soln - np.floor(lp_soln)
            pc_scores = [uppc * upf * dnpc * dnf for uppc, upf, dnpc, dnf in zip(self.up_pc,
                                                                                 up_frac,
                                                                                 self.down_pc,
                                                                                 down_frac)]

            candidates = sorted(range(len(pc_scores)), key=lambda i: -pc_scores[i])
            for var in candidates:
                if np.abs(lp_soln[var] - np.round(lp_soln[var])) > params.EPSILON:
                    branching_var = var
                    break

        print("\t**", branching_var)
        if branching_var is not None:
            # Create branches
            branching_val = lp_soln[branching_var]
            obj_val = self.get_objective_value()
            node_data = get_data(self)
            branches = [(branching_var, consts.LOWER_BOUND, np.floor(branching_val) + 1),
                (branching_var, consts.UPPER_BOUND, np.floor(branching_val))]

            # Create branches
            for branch in branches:
                node_data_clone = node_data.copy()
                node_data_clone['branch_history'] = node_data['branch_history'][:]
                node_data_clone['branch_history'].append(branch)

                self.make_branch(obj_val, variables=[branch], constraints=[], node_data=node_data_clone)

            # Update pseudocosts
            cclone = get_clone(self)
            up_status, up_obj = get_child_obj(self, cclone, branching_var, branching_val, consts.LOWER_BOUND)
            down_status, down_obj = get_child_obj(self, cclone, branching_var, branching_val, consts.UPPER_BOUND)
            # Update up pseudocost
            if up_status == consts.LP_OPTIMAL or up_status == consts.LP_ABORT_IT_LIM:
                up_delta = up_obj - obj_val
                up_delta /= up_frac[branching_var]
                self.up_delta_sum[var] += up_delta
                self.up_count[var] += 1
                self.up_pc[var] = self.up_delta_sum[var] / self.up_count[var]

            # Update down pseudocost
            if down_status == consts.LP_OPTIMAL or down_status == consts.LP_ABORT_IT_LIM:
                down_delta = down_obj - obj_val
                down_delta /= down_frac[branching_var]
                self.down_delta_sum[var] += down_delta
                self.down_count[var] += 1
                self.down_pc[var] = self.down_delta_sum[var] / self.down_count[var]

        # if self.strategy == consts.BS_PC or \
        #     self.strategy == consts.BS_SB or \
        #     self.strategy == consts.BS_SB_PC:
        #     # Get branching candidates based on pseudo costs
        #     candidates = get_candidates(self)
        #     if len(candidates) == 0:
        #         return
        #
        # if self.strategy == consts.BS_SB or \
        #     (self.strategy == consts.BS_SB_PC and self.times_called <= self.THETA):
        #     # Pick variable based on SB score
        #     branching_var = candidates[0]
        #     pass
        #
        # if self.strategy == consts.BS_PC or \
        #     (self.strategy == consts.BS_SB_PC and self.times_called > self.THETA):
        #     # Pick variable based on PC score
        #     branching_var = candidates[0]
        #
        #
        # if self.strategy == consts.BS_PC or \
        #     self.strategy == consts.BS_SB or \
        #     self.strategy == consts.BS_SB_PC:
        #
        #     branching_val = self.get_values(branching_var)
        #     obj_val = self.get_objective_value()
        #     node_data = self.get_data()
        #     branches = [(branching_var, LOWER_BOUND, np.floor(branching_val) + 1),
        #         (branching_var, UPPER_BOUND, np.floor(branching_val))]
        #
        #     for branch in branches:
        #         node_data_clone = node_data.copy()
        #         node_data_clone['branch_history'] = node_data['branch_history'][:]
        #         node_data_clone['branch_history'].append(branch)
        #
        #         self.make_branch(obj_val, variables=[branch], constraints=[], node_data=node_data_clone)

        # if self.times_called == 1:
        #     self.abort()


def solve_instance(path='set_cover.lp',
                   primal_bound=None,
                   timelimit=None,
                   seed=None,
                   test=False,
                   branch_strategy=consts.BS_DEFAULT,
                   theta=params.THETA):
    # Read instance and set default parameters
    c = CPX.Cplex(path)
    set_params(c, primal_bound=primal_bound,
               branch_strategy=branch_strategy,
               timelimit=timelimit,
               seed=seed, test=test)

    log_cb = get_logging_callback(c)

    vsel_cb = c.register_callback(VariableSelectionCallback)
    vsel_cb.c = c
    vsel_cb.var_lst = c.variables.get_names()
    vsel_cb.var_idx_lst = c.variables.get_indices(vsel_cb.var_lst)

    vsel_cb.times_called = 0
    vsel_cb.strategy = branch_strategy

    vsel_cb.up_pc = {v: 0 for v in vsel_cb.var_idx_lst}
    vsel_cb.up_delta_sum = {v: 0 for v in vsel_cb.var_idx_lst}
    vsel_cb.up_count = {v: 0 for v in vsel_cb.var_idx_lst}

    vsel_cb.down_pc = {v: 0 for v in vsel_cb.var_idx_lst}
    vsel_cb.down_delta_sum = {v: 0 for v in vsel_cb.var_idx_lst}
    vsel_cb.down_count = {v: 0 for v in vsel_cb.var_idx_lst}

    # Solve the instance and save stats
    c.solve()

    return c, log_cb
