# import torch.nn as nn
# import torch
from options import get_options
import os
from itertools import product
import json
import sys
import consts

from pathlib import Path

def run(opts):
    baseline_strategy = (
        opts.strategy == consts.BS_DEFAULT
        or opts.strategy == consts.BS_SB
        or opts.strategy == consts.BS_PC
        or opts.strategy == consts.BS_SB_PC 
    )
    online_strategy = (
        opts.strategy == consts.BS_SB_ML_SVMRank
        or opts.strategy == consts.BS_SB_ML_LR
        or opts.strategy == consts.BS_SB_ML_NN
        or opts.strategy == consts.BS_SB_ML_GNN
    )
    
    solve_instance = None
    if baseline_strategy:
        print("* Branching strategy: Baseline")
        from strategy import baseline_solve_instance        
        solve_instance = baseline_solve_instance
    elif online_strategy:
        print("  Branching strategy: Online")
        from strategy import online_solve_instance
        solve_instance = online_solve_instance
    else:
        raise ValueError("Unknown branching strategy")
    
    train_path = Path(opts.train_dataset)
    for f in train_path.glob('*.lp'):
        print("*", str(f))
        solve_instance(path=str(f))
        

if __name__ == "__main__":    
    opts = get_options(sys.argv[1:])
    run(opts)

