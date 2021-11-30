# Dataset types
HOMOGENEOUS = 'HOMOGENEOUS'
HETEROGENEOUS = 'HETEROGENEOUS'
DATASET_TYPE = [
    HOMOGENEOUS,
    HETEROGENEOUS
]

# Run modes
GENERATE_OPTIMAL = 0
BRANCHING = 1
TRAIN_META = 2
MODE = [
    'GENERATE OPTIMAL',
    'BRANCHING',
    'TRAIN_META'
]

# CPLEX sense constant
MINIMIZE = 1

# CPLEX status constant
LP_OPTIMAL = 1
LP_INFEASIBLE = 3
LP_ABORT_IT_LIM = 10
INFEASIBILITY_SCORE = 1e6

# CPLEX branch creation constants
UPPER_BOUND = 'U'
LOWER_BOUND = 'L'

# Branch strategy constants
CPX_DEFAULT = 0
CPX_PC = 1
CPX_SB = 2
BS_SB = 3
BS_PC = 4
BS_SB_PC = 5
BS_SB_ML_SVMRank = 6
BS_SB_ML_NN = 7
STRATEGY = [
    'CPX_DEFAULT',
    'CPX_PC',
    'CPX_SB',
    'SB',
    'PS',
    'SB_PS',
    'SB_SVM_RANK',
    'SB_FFNN',
]

NONE = 0
AVERAGE_MODEL = 1
INCREMENTAL_WARM_START = 2

WARM_START = [
    'NONE',
    'AVERAGE_MODEL',
    'INCREMENTAL_WARM_START'
]
BETA = -1  # Number of instances to solve before warm-starting
