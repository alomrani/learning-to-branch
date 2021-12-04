import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np
from sklearn.svm import SVC

import consts
import params
from featurizer import DynamicFeaturizer, StaticFeaturizer
from models.MLPClassifier import MLPClassifier1 as MLPClassifier
from utils import (get_clone, get_data, get_logging_callback,
                   set_params, solve_as_lp, get_sb_scores, get_candidates)


class VariableSelectionCallback(CPX_CB.BranchCallback):
    def candidate_labels(self, candidate_scores):
        max_score = max(candidate_scores)
        labels = (candidate_scores >= (1 - self.alpha) * max_score).astype(int)

        return labels

    @staticmethod
    def update_bipartite_ranking(node_feat, labels):
        bipartite_rank_labels = []
        K = len(node_feat)
        feature_diff = []
        max_feat = np.argmax(node_feat, axis=0)[None, :]
        node_feat = node_feat / (max_feat + (max_feat == 0).astype(int))
        for i in range(K):
            for j in range(i, K):
                if j != i and labels[j] != labels[i]:
                    feature_diff.append(node_feat[j] - node_feat[i])
                    bipartite_rank_labels.append(labels[j] - labels[i])
                    feature_diff.append(-(node_feat[j] - node_feat[i]))
                    bipartite_rank_labels.append(-(labels[j] - labels[i]))

        return np.asarray(feature_diff), np.asarray(bipartite_rank_labels)

    def __call__(self):
        """For all ML-based strategies, collect branching data for the
        first THETA nodes. For the remaining nodes, select variables based
        on the trained ML model.
        """

        # Find candidates for branching
        candidate_idxs = get_candidates(self)
        if len(candidate_idxs) == 0:
            return

        self.times_called += 1
        # Collect branching data for training ML models
        branching_var_idx = None
        if self.times_called <= self.THETA:
            # print("* Collecting data")
            # Calculate SB scores for branching candidates
            sb_scores, cclone = get_sb_scores(self, candidate_idxs)
            # print('* SB scores', sb_scores)
            sb_scores = np.asarray(sb_scores)
            branching_var_idx = candidate_idxs[np.argmax(sb_scores)]
            # print(f'* Branch variable {branching_var_idx}')

            # Prepare training data
            dynamic = DynamicFeaturizer(self, candidate_idxs, cclone)
            dynamic.features = np.asarray(dynamic.features)
            labels = self.candidate_labels(sb_scores)
            if self.times_called == 1:
                self.node_feat, self.rank_labels = self.update_bipartite_ranking(dynamic.features, labels)
                self.rank_labels = self.rank_labels[:, None]
            else:
                curr_node_feat, curr_rank_labels = self.update_bipartite_ranking(dynamic.features, labels)
                if len(curr_node_feat) > 0:
                    self.node_feat = np.concatenate((self.node_feat, curr_node_feat), axis=0)
                    self.rank_labels = np.concatenate((self.rank_labels, curr_rank_labels[:, None]), axis=0)
            if self.times_called == self.THETA:
                # Train model
                print("* Making dataset")
                feat_diff, rank_labels = self.node_feat, self.rank_labels
                if self.strategy == consts.BS_SB_ML_SVMRank:
                    print("* Training Model")
                    self.model = self.model.fit(feat_diff, rank_labels[:, 0])
                    print("* Done")
                elif self.strategy == consts.BS_SB_ML_NN:
                    print("* Training Model")
                    self.model = self.model.fit(feat_diff, rank_labels[:, 0])
                    print("* Done")

        else:
            # print("* using ML model")
            cclone = get_clone(self)
            # Must be calculated in order to obtain dual prices (dual costs/shadow prices)
            K = len(candidate_idxs)
            status, parent_objval, dual_values = solve_as_lp(cclone)
            self.curr_node_dual_values = np.asarray(dual_values)
            dfobj = DynamicFeaturizer(self, candidate_idxs, cclone)
            dfobj.features = np.asarray(dfobj.features)
            max_feat = dfobj.features.argmax(axis=0)
            dfobj.features = dfobj.features / (max_feat + (max_feat == 0).astype(int))
            X = (np.repeat(dfobj.features[:, None, :], len(dfobj.features), axis=1) - dfobj.features[None, :, :])
            out = self.model.predict(np.reshape(X, (K ** 2, 72))).reshape((K, K, 1))
            branching_var_idx = candidate_idxs[np.argmax(out.sum(axis=1), axis=0).item()]

        assert branching_var_idx is not None
        branch_val = self.get_values(branching_var_idx)
        objval = self.get_objective_value()
        node_data = get_data(self)

        ##################################################################################
        # NOTE: branching_var must be an index of the variable
        branches = [(branching_var_idx, consts.LOWER_BOUND, np.floor(branch_val) + 1),
                    (branching_var_idx, consts.UPPER_BOUND, np.floor(branch_val))]
        ##################################################################################
        for branch in branches:
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            node_data_clone['branch_history'].append(branch)

            self.make_branch(objval, variables=[branch], constraints=[], node_data=node_data_clone)

        # if self.times_called == 5:
        #     self.abort()


def solve_instance(
        path='set_cover.lp',
        cutoff=None,
        timelimit=None,
        seed=None,
        test=True,
        branch_strategy=consts.BS_SB_PC,
        theta=params.THETA,
        max_iterations=50,
        k=params.K,
        alpha=params.ALPHA,
        warm_start_model=None):
    # Read instance and set default parameters
    np.random.seed(seed)
    c = CPX.Cplex(path)
    set_params(c, cutoff=cutoff, timelimit=timelimit,
               seed=seed, test=test, branch_strategy=branch_strategy)

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

    vsel_cb.alpha = alpha
    vsel_cb.K = k
    vsel_cb.num_infeasible_left = np.zeros(num_vars)
    vsel_cb.num_infeasible_right = np.zeros(num_vars)
    vsel_cb.node_feat = []
    vsel_cb.labels = []

    stat_feat = StaticFeaturizer(c, vsel_cb.ordered_var_idx_lst)
    vsel_cb.stat_feat = stat_feat

    if warm_start_model is not None:
        vsel_cb.model = warm_start_model
    elif branch_strategy == consts.BS_SB_ML_SVMRank:
        vsel_cb.model = SVC(gamma='scale', decision_function_shape='ovo', C=0.1, degree=2)
    elif branch_strategy == consts.BS_SB_ML_NN:
        vsel_cb.model = MLPClassifier(learning_rate_init=0.01, n_iter_no_change=100,
                                      max_iter=300, warm_start=True, tol=1e-6)
    # Solve the instance and save stats
    c.solve()

    return c, log_cb, vsel_cb
