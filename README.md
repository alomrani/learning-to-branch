# Meta Learning to Branch in Mixed Integer Programming

`run.py`: File to run

`featurizer/static.py`: static features

`featurizer/dynamic.py`: static features

`options.py`: all file options to run run.py

`strategy/baseline.py`: branching for SB and PC

`strategy/online.py`: branching with learned model


* Usage instructions
    ----------------------------------------------------------------------------------------
    1. For generating mip cutoffs
  
    `python run.py --mode 0 --instance <instance path> --seed <seed>`
    
    2. For branching
    
    `python run.py --mode 1 --instance <instance path> --strategy <strategy_id> --seed <seed>`
    
    3. For learning meta-model
    
    `python run.py --mode 2 --strategy <strategy_id> --seed <seed>`
    
    * Parameters details
    
    1. `<strategy_id>` can be between 0 to 5, where 
    
        0 ==> DEFAULT
    
        1 ==> Strong branching
    
        2 ==> Pseudocost branching
    
        3 ==> Strong(theta) + Pseudocost branching
    
        4 ==> Strong(theta) + SVM Rank
    
        5 ==> Strong(theta) + Feed forward Neural Network
    
    2. `<instance path>` path to instance
  
    3. add option `--parallel` to run many instances in parallel.



### References

1. Khalil, Elias, Pierre Le Bodic, Le Song, George Nemhauser, and Bistra Dilkina. "Learning to branch in mixed integer programming." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 30, no. 1. 2016. 

2. https://github.com/raviagrwl420/variable-selection
