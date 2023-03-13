import os
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script provides results for Figure~3 and Table~5.
Example usage python run.py
The results are saved under ./results folder.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
if __name__=='__main__':
    
    print('Running for Student Datasets')
    nlambda = 25
    for seed in range(10):
        print('Seed {}...'.format(seed))
        os.system('python run_benchmark_regression.py --dataset StudentsMath --seed {} --lambda_max 3 --n_epochs 2000 --lr 1e-3'.format(seed))
        os.system('python run_benchmark_regression.py --dataset StudentsPortugese --seed {} --lambda_max 3 --n_epochs 2000 --lr 1e-3'.format(seed))
        os.system('python run_benchmark_regression.py --dataset StudentsMath --seed {} --lambda_max 3 --n_epochs 2000 --model linearregression --lr 1e-3'.format(seed))
        os.system('python run_benchmark_regression.py --dataset StudentsPortugese --seed {} --lambda_max 3 --n_epochs 2000 --model linearregression --lr 1e-3'.format(seed))
        os.system('python baseline_convex_fair_regression.py --seed {} --fairness individual --dataset StudentsPortugese'.format(seed))
        os.system('python baseline_convex_fair_regression.py --seed {} --fairness individual --dataset StudentsMath'.format(seed))
        os.system('python baseline_convex_fair_regression.py --seed {} --fairness group --dataset StudentsPortugese'.format(seed))
        os.system('python baseline_convex_fair_regression.py --seed {} --fairness group --dataset StudentsMath'.format(seed))

    print('Running for CommunitiesCrime')
    nlambda = 25
    for seed in range(10):
        print('Seed {}...'.format(seed))
        os.system('python run_benchmark_regression.py --dataset CommunitiesCrime --seed {} --n_epochs 1000 --lambda_max 2'.format(seed))
        os.system('python run_benchmark_regression.py --dataset CommunitiesCrime --seed {} --n_epochs 1000 --model linearregression --lambda_max 2'.format(seed))
        os.system('python baseline_convex_fair_regression.py --seed {} --fairness group --dataset CommunitiesCrime'.format(seed))
        os.system('python baseline_convex_fair_regression.py --seed {} --fairness individual --dataset CommunitiesCrime'.format(seed))
