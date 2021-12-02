# Parameters for
# Khalil, E., Le Bodic, P., Song, L., Nemhauser, G., & Dilkina, B. (2016, February).
# Learning to branch in mixed integer programming. In Proceedings of
# the AAAI Conference on Artificial Intelligence (Vol. 30, No. 1).

# Number of nodes to run strong branching for data-collection
THETA = 200
# Maximum number of candidates to calculate strong-branching scores
K = 10
# Threshold to determine integrality of variables
EPSILON = 1e-6
# Cutoff to decide variables with label 1
ALPHA = 0.2

SEEDS = [3]

#################################################################
# Additional parameters for meta-learning
# Number of instances to solve before warm-starting
BETA = 0
# Number of training nodes used for learning and instance specific
# model after BETA instances
THETA2 = 0
