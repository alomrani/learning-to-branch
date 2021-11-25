from operator import itemgetter
from platform import node
import time

import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np
from sklearn.svm import SVC
from sklearn.utils.multiclass import check_classification_targets
from models.MLPClassifier import MLPClassifier1 as MLPClassifier
import consts
import params
from featurizer import DynamicFeaturizer, StaticFeaturizer
import strategy
from utils import (get_branch_solution, get_clone, get_data, get_logging_callback,
                   set_params, solve_as_lp, get_sb_scores, get_candidates)


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def get_pseudocosts_score(self, pseudocosts, soln, floor_soln, ceil_soln):
        """Assumption: p[0] == Down pseudocost, p[1] == Up pseudocost"""
        pseudocosts_score = [0] * len(pseudocosts)
        for i, (p, s, f, c, vid) in enumerate(zip(pseudocosts, soln, floor_soln, ceil_soln, self.var_idx_lst)):
            pseudocosts_score[i] = ((p[0] * (s - f)) * (p[1] * (c - s)), vid)
        pseudocosts_score = sorted(pseudocosts_score, key=itemgetter(0), reverse=True)

        return pseudocosts_score

    # def get_candidates(self):
    #     soln = self.get_values(self.var_idx_lst)
    #     candidate_soln, candidate_idx = [], []
    #     candidate_soln_idx_map = [(s, i) for s, i in zip(soln, self.var_idx_lst) if
    #                               not (abs(s - round(s)) <= self.EPSILON)]
    #     soln = [si[0] for si in candidate_soln_idx_map]
    #     vidx = [si[1] for si in candidate_soln_idx_map]

    #     pseudocosts = self.get_pseudo_costs(vidx)
    #     floor_soln = np.floor(soln)
    #     ceil_soln = np.ceil(soln)

    #     pseudocosts_score = self.get_pseudocosts_score(pseudocosts, soln, floor_soln, ceil_soln)
    #     max_k = self.K if len(pseudocosts_score) >= self.K else len(pseudocosts_score)
    #     pc_scores = [i[0] for i in pseudocosts_score[: max_k]]
    #     candidates = [i[1] for i in pseudocosts_score[: max_k]]

    #     return candidates, pc_scores

    # def get_strong_branching_score(self, candidates):
    #     cclone = get_clone(self)
    #     status, parent_objval, dual_values = solve_as_lp(cclone)

    #     self.curr_node_dual_values = np.array(dual_values)
    #     sb_scores = []
    #     for cand in candidates:
    #         status_lower, lower_objective = get_branch_solution(self, cclone, cand, consts.LOWER_BOUND)
    #         status_upper, upper_objective = get_branch_solution(self, cclone, cand, consts.UPPER_BOUND)

    #         delta_lower = max(lower_objective - parent_objval, params.EPSILON)
    #         delta_upper = max(upper_objective - parent_objval, params.EPSILON)

    #         sb_score = delta_lower * delta_upper
    #         sb_scores.append(sb_score)

    #         if status_lower != consts.LP_OPTIMAL:
    #             self.num_infeasible_left[cand] += 1
    #         if status_upper != consts.LP_OPTIMAL:
    #             self.num_infeasible_right[cand] += 1
    #     return np.asarray(sb_scores), cclone

    def candidate_labels(self, candidate_scores):
        max_score = max(candidate_scores)

        labels = (candidate_scores >= (1 - self.alpha) * max_score).astype(int)
        return labels

    def bipartite_ranking(self, labels):
        bipartite_rank_labels = []
        feature_diff = []
        node_feat = self.node_feat.reshape((self.THETA, self.K, 72))
        max_feat = np.argmax(node_feat, axis=1)[:, None, :]
        node_feat = node_feat / (max_feat + (max_feat == 0).astype(int))
        for i in range(self.THETA):
            for j in range(self.K):
                for k in range(j, self.K):
                    if j != k and labels[i, j] != labels[i, k]:
                        feature_diff.append(node_feat[i, j] - node_feat[i, k])
                        bipartite_rank_labels.append(labels[i, j] - labels[i, k])
                        feature_diff.append(-(node_feat[i, j] - node_feat[i, k]))
                        bipartite_rank_labels.append(-(labels[i, j] - labels[i, k]))
        return np.asarray(feature_diff), np.asarray(bipartite_rank_labels)

    def __call__(self):

        # For all ML-based strategies, collect branching data for the first THETA nodes.
        # For the remaining nodes, select variables based on the trained ML model.

        # Get branching candidates based on pseudo costs
        # candidates, ps_scores = self.get_candidates()
        pseudocosts = self.get_pseudo_costs()
        values = self.get_values()
        candidates = get_candidates(pseudocosts, values, self.strategy)
        if len(candidates) == 0:
            return
        self.times_called += 1
        # print(candidates)
        # Collect branching data for training ML models
        branch_var = None
        if self.times_called <= self.THETA:
            # print("* Collecting data")      
            # Calculate SB scores for branching candidates
            sb_scores, cclone = get_sb_scores(self, candidates)
            # print('* SB scores', sb_scores)
            sb_scores = np.asarray(sb_scores)
            branch_var = candidates[np.argmax(sb_scores)]
            # print(f'* Branch variable {branch_var}')

            # Prepare training data
            dynamic = DynamicFeaturizer(self, candidates, cclone)
            dynamic.features = np.asarray(dynamic.features)
            if type(self.node_feat) is list:
                self.node_feat = dynamic.features[None, :, :]
                self.labels = self.candidate_labels(sb_scores)[None, :]
            else:
                self.node_feat = np.concatenate((self.node_feat, dynamic.features[None, :, :]), axis=0)
                self.labels = np.concatenate((self.labels, self.candidate_labels(sb_scores)[None, :]), axis=0)
            if self.times_called == self.THETA:
                # Train model
                print("* Making dataset")
                feat_diff, rank_labels = self.bipartite_ranking(self.labels)
                if self.strategy == consts.BS_SB_ML_SVMRank:
                    print("* Training Model")
                    self.model = self.model.fit(feat_diff, rank_labels)
                    print("* Done")
                elif self.strategy == consts.BS_SB_ML_NN:
                    print("* Training Model")
                    self.model = self.model.fit(feat_diff, rank_labels)
                    print("* Done")

        else:
            # print("* using ML model")
            cclone = get_clone(self)
            # Must be calculated in order to obtain dual prices (dual costs/shadow prices)
            K = len(candidates)
            status, parent_objval, dual_values = solve_as_lp(cclone)
            self.curr_node_dual_values = np.asarray(dual_values)
            dfobj = DynamicFeaturizer(self, candidates, cclone)
            dfobj.features = np.asarray(dfobj.features)
            max_feat = dfobj.features.argmax(axis=0)
            dfobj.features = dfobj.features / (max_feat + (max_feat == 0).astype(int))
            X = (np.repeat(dfobj.features[:, None, :], len(dfobj.features), axis=1) - dfobj.features[None, :, :])
            out = self.model.predict(np.reshape(X, (K ** 2, 72))).reshape((K, K, 1))
            branch_var = candidates[np.argmax(out.sum(axis=1), axis=0).item()]
        assert branch_var is not None
        branch_val = self.get_values(branch_var)
        objval = self.get_objective_value()
        node_data = get_data(self)
        branches = [(branch_var, self.LOWER_BOUND, np.floor(branch_val) + 1),
                    (branch_var, self.UPPER_BOUND, np.floor(branch_val))]

        for branch in branches:
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            node_data_clone['branch_history'].append(branch)

            self.make_branch(objval, variables=[branch], constraints=[], node_data=node_data_clone)

        # if self.times_called == 5:
        #     self.abort()


def solve_instance(path='set_cover.lp',
                   primal_bound=None,
                   timelimit=None,
                   seed=None,
                   test=True,
                   branch_strategy=consts.BS_SB_PC,
                   upper_bound=consts.UPPER_BOUND,
                   lower_bound=consts.LOWER_BOUND,
                   theta=params.THETA,
                   k=params.K,
                   alpha=params.ALPHA,
                   epsilon=params.EPSILON,
                   warm_start_model=None):
    # Read instance and set default parameters
    np.random.seed(seed)
    c = CPX.Cplex(path)
    set_params(c, primal_bound=primal_bound, timelimit=timelimit,
               seed=seed, test=test, branch_strategy=branch_strategy)

    stat_feat = StaticFeaturizer(c)
    var_lst = c.variables.get_names()
    var_idx_lst = c.variables.get_indices(var_lst)

    log_cb = get_logging_callback(c)

    vsel_cb = c.register_callback(VariableSelectionCallback)
    vsel_cb.c = c
    vsel_cb.var_lst = var_lst
    vsel_cb.var_idx_lst = var_idx_lst
    vsel_cb.stat_feat = stat_feat
    vsel_cb.strategy = branch_strategy
    vsel_cb.times_called = 0
    vsel_cb.THETA = theta
    vsel_cb.alpha = alpha
    vsel_cb.K = k
    vsel_cb.EPSILON = epsilon
    vsel_cb.UPPER_BOUND = upper_bound
    vsel_cb.LOWER_BOUND = lower_bound
    vsel_cb.num_infeasible_left = np.zeros(len(var_idx_lst))
    vsel_cb.num_infeasible_right = np.zeros(len(var_idx_lst))
    vsel_cb.node_feat = []
    vsel_cb.labels = []
    if warm_start_model is not None:
        vsel_cb.model = warm_start_model
    elif branch_strategy == consts.BS_SB_ML_SVMRank:
        vsel_cb.model = SVC(gamma='scale', decision_function_shape='ovo', C=0.1, degree=2, kernel='linear')
    elif branch_strategy == consts.BS_SB_ML_NN:
        vsel_cb.model = MLPClassifier(learning_rate_init=0.01, n_iter_no_change=50, max_iter=300, warm_start=True)
    # Solve the instance and save stats
    c.solve()

    return c, log_cb, vsel_cb.model
