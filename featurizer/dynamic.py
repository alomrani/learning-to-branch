from math import floor
from operator import itemgetter

import numpy as np
from scipy.sparse import csr_matrix


class DynamicFeaturizer:
    def __init__(self, branch_instance, candidates, cclone):

        static_features = branch_instance.stat_feat
        # Part 1: Slack and ceil distances
        self.values = np.array(branch_instance.get_values()).reshape(-1, 1)
        self.values = self.values[candidates]  # Filter by candidates

        # 1. Min of slack and ceil
        ceil = np.ceil(self.values)
        floor = np.floor(self.values)
        fractionality = np.minimum(self.values - floor, ceil - self.values)

        # 2. Distance from ceil
        dist_ceil = ceil - self.values

        # Add 1, 2 to features
        self.features = np.c_[fractionality, dist_ceil]


        # Part 2: Pseudocosts

        # 3. Upwards and downwards pseudocosts weighted by fractionality
        self.pseudocosts = np.array(branch_instance.get_pseudo_costs())
        self.pseudocosts = self.pseudocosts[candidates]
        up_down_pc = self.pseudocosts * fractionality

        # 4. Sum of pseudocosts weighted by fractionality
        sum_pc = np.sum(self.pseudocosts, axis=1).reshape(-1, 1) * fractionality

        # 5. Ratio of pseudocosts weighted by fractionality
        ratio_pc = (self.pseudocosts[:, 0] / self.pseudocosts[:, 1]).reshape(-1, 1) * fractionality
        ratio_pc[np.isnan(ratio_pc)] = 0
        ratio_pc[np.isinf(ratio_pc)] = 0

        # 6. Prod of pseudocosts weighted by fractionality
        prod_pc = np.prod(self.pseudocosts, axis=1).reshape(-1, 1) * fractionality

        # Add 3, 4, 5, 6 to features
        self.features = np.c_[self.features, up_down_pc, sum_pc, ratio_pc, prod_pc]

        # print('1, 2', self.features.shape)

        # Part 3: Infeasibility statistics
        num_lower_infeasible = branch_instance.num_infeasible_left[np.array(candidates)][:, None]
        num_upper_infeasible = branch_instance.num_infeasible_right[np.array(candidates)][:, None]

        fraction_infeasible_lower = num_lower_infeasible / branch_instance.times_called
        fraction_infeasible_upper = num_upper_infeasible / branch_instance.times_called

        self.features = np.c_[
            self.features, num_lower_infeasible, num_upper_infeasible, fraction_infeasible_lower, fraction_infeasible_upper]
        # Part 4: Stats. for constraint degrees
        not_set_variables = (np.array(cclone.variables.get_lower_bounds()) != np.array(cclone.variables.get_upper_bounds()))
        self.matrix = static_features.matrix.multiply(csr_matrix(not_set_variables[None, :]))
        non_zeros = self.matrix != 0
        num_const_for_var = np.transpose(non_zeros.sum(0))
        num_var_for_const = non_zeros.sum(1)
        degree_matrix = non_zeros.multiply(csr_matrix(num_var_for_const)).todense()

        # Mean of degrees
        mean_degrees = np.transpose(np.mean(degree_matrix, axis=0))[candidates, :]

        # Stdev of degrees
        std_degrees = np.transpose(np.std(degree_matrix, axis=0))[candidates, :]

        # Min of degrees
        min_degrees = np.transpose(np.min(degree_matrix, axis=0))[candidates, :]

        # Max of degrees
        max_degrees = np.transpose(np.max(degree_matrix, axis=0))[candidates, :]
        
        # Mean Ratio static to Dynamic
        mean_degrees_c = mean_degrees.copy()
        mean_degrees_c[mean_degrees_c == 0.] = 1.
        mean_degrees_ratio = static_features.features[candidates, 4] / mean_degrees_c

        # Min Ratio static to Dynamic
        min_degrees_c = min_degrees.copy()
        min_degrees_c[min_degrees_c == 0.] = 1.
        min_degrees_ratio = static_features.features[candidates, 6] / min_degrees_c

        # Max Ratio static to Dynamic
        max_degrees_c = max_degrees.copy()
        max_degrees_c[max_degrees_c == 0.] = 1.
        max_degrees_ratio = static_features.features[candidates, 7] / max_degrees_c

        # Add to features
        self.features = np.c_[
            self.features, mean_degrees, std_degrees, min_degrees, max_degrees, mean_degrees_ratio, min_degrees_ratio, max_degrees_ratio]

        # Part 5: Min/max ratios of constraint coeffs to RHS
        rhs = static_features.rhs.reshape(-1, 1)
        pos_rhs = rhs[rhs > 0]
        neg_rhs = rhs[rhs < 0]

        mat = static_features.matrix.todense()
        candidate_matrix = static_features.matrix[:, candidates].todense()
        pos_ratio_matrix = np.divide(candidate_matrix[(rhs > 0).ravel(), :], pos_rhs.reshape(-1, 1))
        pos_ratio_matrix = pos_ratio_matrix if pos_ratio_matrix.size else np.zeros((1, candidate_matrix.shape[1]))
        neg_ratio_matrix = np.divide(candidate_matrix[(rhs < 0).ravel(), :], neg_rhs.reshape(-1, 1))
        neg_ratio_matrix = neg_ratio_matrix if neg_ratio_matrix.size else np.zeros((1, candidate_matrix.shape[1]))

        # 7. Min ratio for positive RHS
        min_ratio_pos = np.transpose(np.min(pos_ratio_matrix, axis=0))

        # 8. Max ratio for positive RHS
        max_ratio_pos = np.transpose(np.max(pos_ratio_matrix, axis=0))

        # 9. Min ratio for negative RHS
        min_ratio_neg = np.transpose(np.min(neg_ratio_matrix, axis=0))

        # 10. Max ratio for negative RHS
        max_ratio_neg = np.transpose(np.max(neg_ratio_matrix, axis=0))

        # Add 7, 8, 9, 10 to features
        self.features = np.c_[self.features, min_ratio_pos, max_ratio_pos, min_ratio_neg, max_ratio_neg]
        # print('1, 2, 5', self.features.shape)

        # Part 6: Min/max for one-to-all coefficient ratios
        pos_coeff_matrix = static_features.pos_coeff_matrix[:, candidates]
        neg_coeff_matrix = static_features.neg_coeff_matrix[:, candidates]

        sum_pos_coeffs = static_features.sum_pos_coeffs
        sum_pos_coeffs = sum_pos_coeffs + (sum_pos_coeffs == 0.).astype(int)
        sum_neg_coeffs = static_features.sum_neg_coeffs
        sum_neg_coeffs = sum_neg_coeffs + (sum_neg_coeffs == 0.).astype(int)

        pos_pos_ratio_matrix = pos_coeff_matrix.todense() / sum_pos_coeffs
        pos_neg_ratio_matrix = pos_coeff_matrix.todense() / sum_neg_coeffs
        neg_neg_ratio_matrix = neg_coeff_matrix.todense() / sum_neg_coeffs
        neg_pos_ratio_matrix = neg_coeff_matrix.todense() / sum_pos_coeffs

        pos_pos_ratio_min = np.transpose(np.min(pos_pos_ratio_matrix, axis=0))
        pos_pos_ratio_max = np.transpose(np.max(pos_pos_ratio_matrix, axis=0))
        pos_neg_ratio_min = np.transpose(np.min(pos_neg_ratio_matrix, axis=0))
        pos_neg_ratio_max = np.transpose(np.max(pos_neg_ratio_matrix, axis=0))
        neg_neg_ratio_min = np.transpose(np.min(neg_neg_ratio_matrix, axis=0))
        neg_neg_ratio_max = np.transpose(np.max(neg_neg_ratio_matrix, axis=0))
        neg_pos_ratio_min = np.transpose(np.min(neg_pos_ratio_matrix, axis=0))
        neg_pos_ratio_max = np.transpose(np.max(neg_pos_ratio_matrix, axis=0))

        self.features = np.c_[self.features, pos_pos_ratio_min, pos_pos_ratio_max, pos_neg_ratio_min, pos_neg_ratio_max,
                              neg_neg_ratio_min, neg_neg_ratio_max, neg_pos_ratio_min, neg_pos_ratio_max]

        # print('1, 2, 5, 6', self.features.shape)

        # Part 7: Stats for active constraints
        ## TODO: CHECK IF STATS ARE OVER ABSOLUTE VALUE OF CONSTRAINSTS COEFFICIENTS
        slacks = np.array(branch_instance.get_linear_slacks())
        active_constraints = slacks == 0

        active_matrix = static_features.matrix[active_constraints, :]
        active_matrix = active_matrix[:, candidates].todense()
        count_active_matrix = active_matrix != 0

        # Unit weighting
        unit_sum = np.transpose(np.sum(active_matrix, axis=0))
        unit_mean = np.transpose(np.mean(active_matrix, axis=0))
        unit_std = np.transpose(np.std(active_matrix, axis=0))
        unit_min = np.transpose(np.min(active_matrix, axis=0))
        unit_max = np.transpose(np.max(active_matrix, axis=0))
        unit_count = np.transpose(np.sum(count_active_matrix, axis=0))

        # Add unit weighting features
        self.features = np.c_[self.features, unit_sum, unit_mean, unit_std, unit_min, unit_max, unit_count]
        # print('1, 2, 5, 6, 7a', self.features.shape)

        # Inverse sum all weighting
        sum_coeff = static_features.sum_coeffs[active_constraints]
        sum_coeff = sum_coeff + (sum_coeff == 0.).astype(int)
        inverse_sum_all = 1 / sum_coeff
        inverse_sum_all_matrix = np.multiply(active_matrix, inverse_sum_all)
        count_inverse_sum_all_matrix = np.multiply(count_active_matrix, inverse_sum_all)

        inv_sum_all_sum = np.transpose(np.sum(inverse_sum_all_matrix, axis=0))
        inv_sum_all_mean = np.transpose(np.mean(inverse_sum_all_matrix, axis=0))
        inv_sum_all_std = np.transpose(np.std(inverse_sum_all_matrix, axis=0))
        inv_sum_all_min = np.transpose(np.min(inverse_sum_all_matrix, axis=0))
        inv_sum_all_max = np.transpose(np.max(inverse_sum_all_matrix, axis=0))
        inv_sum_all_count = np.transpose(np.sum(count_inverse_sum_all_matrix, axis=0))

        # Add inverse sum all weighting features
        self.features = np.c_[
            self.features, inv_sum_all_sum, inv_sum_all_mean, inv_sum_all_std, inv_sum_all_min, inv_sum_all_max, inv_sum_all_count]
        # print('1, 2, 5, 6, 7b', self.features.shape)

        # Inverse sum candidate weighting
        sum_active = np.sum(active_matrix, axis=1)
        sum_active = sum_active + (sum_active == 0.).astype(int)
        inverse_sum_candidate = 1 / sum_active
        inverse_sum_candidate_matrix = np.multiply(active_matrix, inverse_sum_candidate)
        count_inverse_sum_candidate_matrix = np.multiply(count_active_matrix, inverse_sum_candidate)

        inv_sum_candidate_sum = np.transpose(np.sum(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_mean = np.transpose(np.mean(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_std = np.transpose(np.std(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_min = np.transpose(np.min(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_max = np.transpose(np.max(inverse_sum_candidate_matrix, axis=0))
        inv_sum_candidate_count = np.transpose(np.sum(count_inverse_sum_candidate_matrix, axis=0))

        # Dual Cost weighting
        dual_values = np.array(branch_instance.curr_node_dual_values[np.array(active_constraints)])[:, None]
        active_matrix = np.asarray(active_matrix)
        dual_sum = np.sum(active_matrix * dual_values, axis=0)[:, None]
        dual_mean = np.mean(active_matrix * dual_values, axis=0)[:, None]
        dual_std = np.std(active_matrix * dual_values, axis=0)[:, None]
        dual_min = np.min(active_matrix * dual_values, axis=0)[:, None]
        dual_max = np.max(active_matrix * dual_values, axis=0)[:, None]
        dual_count = np.sum(np.asarray(count_active_matrix) * dual_values, axis=0)[:, None]

        self.features = np.c_[
            static_features.features[candidates, :],
            self.features, inv_sum_candidate_sum,
            inv_sum_candidate_mean,
            inv_sum_candidate_std,
            inv_sum_candidate_min,
            inv_sum_candidate_max,
            inv_sum_candidate_count,
            dual_sum,
            dual_mean,
            dual_std,
            dual_min,
            dual_max,
            dual_count
        ]

        
        # print(self.features.shape)
