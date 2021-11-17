# import torch.nn as nn
# import torch
from options import get_options
import os
from itertools import product
import json
import sys
import consts
import pickle as pkl
from pathlib import Path
from utils import disable_output

def run(opts):
    if opts.mode == consts.GENERATE_OPTIMAL:
        import cplex as CPX
        from params import SEEDS
        
        train_path = Path(opts.train_dataset)
        output_path = Path(opts.output_dir)
        output_path = output_path / "train"
        output_path.mkdir(parents=True, exist_ok=True)
        
        opt_dict = {}
        for f in train_path.glob('*.lp'):
            print("*", str(f))
            # print(f.name)
            c = CPX.Cplex(str(f))                
            disable_output(c)
            c.solve()
            if c.solution.get_status() == c.solution.status.MIP_optimal:
                opt_dict[f.name] = c.solution.get_objective_value()
                
        pkl.dump(opt_dict, open(output_path / 'optimal_obj.pkl', 'wb'))
        # od = pkl.load(open(output_path / 'optimal_obj.pkl', 'rb'))
        # print(od)
        
    elif opts.mode == consts.BRANCHING:
        assert 0 <= opts.strategy <= len(consts.STRATEGY), "Unknown branching strategy"
        print(f'* Branching strategy: {consts.STRATEGY[opts.strategy]}')
        
        baseline_strategy = (
            opts.strategy == consts.BS_DEFAULT
            or opts.strategy == consts.BS_SB
            or opts.strategy == consts.BS_PC
            or opts.strategy == consts.BS_SB_PC 
        )
        
        solve_instance = None
        if baseline_strategy:
            from strategy import baseline_solve_instance        
            solve_instance = baseline_solve_instance
        else:
            from strategy import online_solve_instance
            solve_instance = online_solve_instance
        
        train_path = Path(opts.train_dataset)
        for f in train_path.glob('*.lp'):
            print("*", str(f))
            c = solve_instance(path=str(f))
            if c.solution.status == c.solution.status.MIP_optimal:
                pass
            

if __name__ == "__main__":    
    opts = get_options(sys.argv[1:])
    run(opts)

