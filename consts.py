# Run modes
GENERATE_OPTIMAL = 1
BRANCHING = 2
MODE = [
    'GENERATE OPTIMAL',
    'BRANCHING'
]

# CPLEX sense constant
MINIMIZE = 1

# CPLEX status constant
OPTIMAL = 1
INFEASIBILITY = 1e-6

# CPLEX branch creation constants
UPPER_BOUND = 'U'
LOWER_BOUND = 'L'

# Branch strategy constants
BS_DEFAULT = 1
BS_SB = 2
BS_PC = 3
BS_SB_PC = 4
BS_SB_ML_SVMRank = 5
BS_SB_ML_LR = 6
BS_SB_ML_NN = 7
BS_SB_ML_GNN = 8

STRATEGY = [
    'DEFAULT',
    'STRONG BRANCHING',
    'PSEUDOCOST BRANCHING',
    'STRONG+PSEUDOCOST BRANCHING',
    'SVM RANK',
    'LINEAR REGRESSION',
    'FEEDFORWARD NEURAL NETWORK',
    'GRAPH NEURAL NETWORK'
]
