import pickle as pkl
import sys
from pathlib import Path

import consts
from options import get_options
from utils import disable_output


def run(opts):
    # Load instance
    # /scratch/rahulpat/setcover/train/1000_1000/1000_1000_0.lp
    instance_path = Path(opts.instance)
    assert instance_path.exists(), "Instance not found!"

    # Set output path
    # /scratch/rahulpat/setcover/output/
    output_dir_path = instance_path.parent.parent.parent / "output"
    # /scratch/rahulpat/setcover/output/train
    output_dir_path = output_dir_path / instance_path.parent.parent.stem
    # /scratch/rahulpat/setcover/output/train/1000_1000/
    output_dir_path = output_dir_path / instance_path.parent.stem

    print(f'* Run mode: {consts.MODE[opts.mode]}')
    if opts.mode == consts.GENERATE_OPTIMAL:
        valid_extensions = ['.lp', '.mps', '.mps.gz']
        assert instance_path.suffix in valid_extensions, "Invalid instance file format!"

        # Solve instance
        import cplex as CPX
        c = CPX.Cplex(str(instance_path))
        c.parameters.timelimit.set(opts.timelimit)
        disable_output(c)
        c.solve()

        # Fetch optimal objective value
        opt_dict = {}
        objective_value = None
        if c.solution.get_status() == c.solution.status.MIP_optimal:
            objective_value = c.solution.get_objective_value()
        opt_dict[instance_path.name] = objective_value

        # Save result
        output_dir_path = output_dir_path.joinpath("optimal_obj")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path.joinpath(instance_path.stem + ".pkl")
        pkl.dump(opt_dict, open(output_path, 'wb'))

    elif opts.mode == consts.BRANCHING:
        assert 0 <= opts.strategy <= len(consts.STRATEGY), "Unknown branching strategy"
        print(f'* Branching strategy: {consts.STRATEGY[opts.strategy]}')
        print(f"* File: {str(instance_path)}\n* Seed: {opts.seed}")

        # Load relevant solve_instance()
        baseline_strategy = (
                opts.strategy == consts.BS_DEFAULT
                or opts.strategy == consts.BS_SB
                or opts.strategy == consts.BS_PC
                or opts.strategy == consts.BS_SB_PC
        )
        if baseline_strategy:
            from strategy import baseline_solve_instance
            solve_instance = baseline_solve_instance
        else:
            from strategy import online_solve_instance
            solve_instance = online_solve_instance

        # Check if optimal solution exists to provide as primal bound
        primal_bound = 1e6
        optimal_obj_path = output_dir_path.joinpath(f"optimal_obj/{instance_path.stem}.pkl")
        print(f"* Checking optimal objective pickle at {optimal_obj_path}...")
        if optimal_obj_path.exists():
            opt_dict = pkl.load(open(optimal_obj_path, 'rb'))
            if instance_path.name in opt_dict and opt_dict[instance_path.name] is not None:
                primal_bound = opt_dict[instance_path.name]
                print(f"* Primal bound: {primal_bound}")
            else:
                print(f"* Instance primal bound not found...")
        else:
            print("* Warning: Optimal objective pickle not found. Can't set primal bound.")

        print("* Starting the solve...")
        c, log_cb = solve_instance(path=str(instance_path),
                                   primal_bound=primal_bound,
                                   timelimit=opts.timelimit,
                                   branch_strategy=opts.strategy,
                                   seed=opts.seed,
                                   test=False)

        solve_status_id = c.solution.get_status()
        solve_status_verbose = c.solution.status[c.solution.get_status()]
        # Get the number of nodes and time taken to solve the instance
        num_nodes, total_time = None, None
        if solve_status_id == c.solution.status.MIP_optimal:
            num_nodes = c.solution.progress.get_num_nodes_processed()
            total_time = log_cb.total_time

        print(f"\n\tS: {solve_status_verbose}, T: {total_time}, N:{num_nodes}\n")

        # Prepare result
        result_dict = {str(instance_path): {'status': solve_status_id,
                                            'status_verbose': solve_status_verbose,
                                            'total_time': total_time,
                                            'num_nodes': num_nodes}}

        # /scratch/rahulpat/setcover/output/train/1000_1000/SB_PC
        output_dir_path = output_dir_path / consts.STRATEGY[opts.strategy]
        output_dir_path.mkdir(parents=True, exist_ok=True)
        # /scratch/rahulpat/setcover/output/train/1000_1000/SB_PC/1000_1000_0.pkl
        output_path = output_dir_path.joinpath(str(instance_path.stem) + ".pkl")

        # Save results
        pkl.dump(result_dict, open(output_path, 'wb'))
        print(f"* Output file path: {output_path}")


if __name__ == "__main__":
    """
    * Usage instructions
    ----------------------------------------------------------------------------------------
    1. For generating optimal solutions
    python run.py --mode 0 --instance <instance path> --seed <seed>
    
    2. For branching
    python run.py --mode 1 --instance <instance path> --strategy <strategy_id> --seed <seed>
    
    * Parameters details
    ----------------------------------------------------------------------------------------     
    1. <instance_path> must be in the following format
        <some_directory>/<problem_type>/<split_type>/<problem_size>/<instance_name>
            <some_directory>    /home/rahul
            <problem_type>      /setcover
            <split_type>        /train 
            <problem_size>      /1000_1000
            <instance_name>     /1000_1000_0.lp
            
    2. <strategy_id> can be between 0 to 6, where 
        0 ==> DEFAULT
        1 ==> Strong branching
        2 ==> Pseudocost branching
        3 ==> Strong(theta) + Pseudocost branching
        4 ==> Strong(theta) + SVM Rank
        5 ==> Strong(theta) + Linear Regression
        6 ==> Strong(theta) + Feed forward Neural Network
        

    """
    opts = get_options(sys.argv[1:])
    run(opts)
