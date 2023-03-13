import os
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script provides results for Table~7.
Example usage python run.py
The results are saved under ./results folder.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
if __name__ == '__main__':
    # experiments for energy distance
    nlambda = 25
    for dataset in ['CommunitiesCrimeClassification', 'Compas', 'Adult', 'Drug']:
        print(dataset)
        for seed in range(10):
            print('Running for seed {}'.format(seed))
            os.system('python run_benchmark1.py --dataset {} --seed {} --a_inside_x True --nlambda {} --model linear'.format(dataset, seed, nlambda))
            os.system('python zafar_classification.py --dataset {} --seed {} --nlambda {}'.format(dataset, seed, nlambda))
            if not dataset == 'Adult':
                os.system('python MMD_fair_run.py --dataset {} --nlambda {} --a_inside_x True --seed {}'.format(dataset, nlambda, seed))
            os.system('python run_benchmark.py --dataset {} --seed {} --a_inside_x True --nlambda {}'.format(dataset, seed, nlambda))
            os.system('python fair_KDE.py --dataset {} --seed {} --nlambda {}'.format(dataset, seed, nlambda))
