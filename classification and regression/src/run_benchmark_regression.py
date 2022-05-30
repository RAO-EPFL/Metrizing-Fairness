import models
import fairness_metrics
import benchmark
import data_loader
import pickle

import argparse
import pandas as pd
import numpy as np
import time
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides implementatino of MFL
An example usage 
python run_benchmark_regression.py --dataset {} --seed {} --nlambda {}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def run(args):
    # act on experiment parameters:
    data_loader.set_seed(args.seed)
    Model = models.LinearRegression if args.model=='linearregression' else models.NeuralNetwork
    fair_loss = fairness_metrics.energy_distance
    lambda_candidates = np.logspace(args.lambda_min, args.lambda_max, num=args.nlambda)
    train_test_split_fin = 0
    lr = args.lr
    lr_decay = 1
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    
    if args.dataset == 'CommunitiesCrime':
        ds = data_loader.CommunitiesCrime()
    if args.dataset == 'BarPass':
        ds = data_loader.BarPass()
    if args.dataset == 'StudentsMath':
        ds = data_loader.StudentPerformance(subject='Math')
    if args.dataset == 'StudentsPortugese':
        ds = data_loader.StudentPerformance(subject='Portugese')

    logfairloss = fair_loss
    ds.split_test()

    k = ds.get_k() # Dimension

    # metrics to evaluate
    metrics = {
        'statistical_parity' : fairness_metrics.statistical_parity,
        'bounded_group_loss_L1' : lambda y1_hat, y2_hat, y1, y2: fairness_metrics.bounded_group_loss(y1_hat, y2_hat, y1, y2, loss='L1'),
        'bounded_group_loss_L2' : fairness_metrics.bounded_group_loss,
        'group_fair_expect' : fairness_metrics.group_fair_expect,
        'l1_dist' : lambda y1_hat, y2_hat, y1, y2: fairness_metrics.lp_dist(y1_hat, y2_hat, y1, y2, p=1),
        'l2_dist' : lambda y1_hat, y2_hat, y1, y2: fairness_metrics.lp_dist(y1_hat, y2_hat, y1, y2, p=2),
        'MSE' : fairness_metrics.MSE,
        'MAE' : fairness_metrics.MAE,
        'R2' : fairness_metrics.R2
    }

    # storage of results
    results_train = []
    results_test = []

    # run the test for various lambdas
    for lambda_ in lambda_candidates:
        print('Training Our method, for lambda_: {}/{}, seed:{}'.format(lambda_, args.nlambda, args.seed))

        model = Model(k)
        train_metrics, test_metrics = benchmark.train_test_fair_learning_regression(ds=ds, 
                                                                 model=model,
                                                                 fair_loss=fair_loss, 
                                                                 lr=lr,
                                                                 batch_size=batch_size, 
                                                                 N_epochs=n_epochs, 
                                                                 lambda_=lambda_, 
                                                                 metrics=metrics, 
                                                                 lr_decay=lr_decay,
                                                                 psi=None, plot_convergence=args.plot_convergence, logfairloss=logfairloss, weight_decay=args.weight_decay, train_test_split_fin=train_test_split_fin)
     
        train_metrics['lambda_'] = lambda_
        test_metrics['lambda_'] = lambda_
        results_train.append(train_metrics)
        results_test.append(test_metrics)
    
    # save the results
    df_train = pd.DataFrame(data=results_train)
    df_test = pd.DataFrame(data=results_test)
    df_train.to_csv('results/NN_energy_regression/{}_{}_train_{}.csv'.format(args.dataset, \
        args.model, args.seed))

    df_test.to_csv('results/NN_energy_regression/{}_{}_test_{}.csv'.format(args.dataset, \
        args.model, args.seed))
    
    
    PARAMS = {'dataset':args.dataset, 
              'batch_size':batch_size,
               'lr':lr, 'epochs':n_epochs,
              'seed':args.seed, 
              'nlambda': args.nlambda, 
              'lambda_min':args.lambda_min,
              'lambda_max':args.lambda_max,
             'algorihtm':'adam', 
              'model_details':model.state_dict,
              'L':'MSE',
              'fair_loss':'Energy',
              'lr_decay':lr_decay
             }
    with open('results/NN_energy_regression/{}_{}_{}.pkl'.format(args.dataset, args.model, args.seed), 'wb') as f:
            pickle.dump({**PARAMS}, f, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--seed', default=0, help='Randomness seed', type=int)
    parser.add_argument('--model', default='NN', choices=['linearregression', 'NN'], help='Regression Model')
    parser.add_argument('--lambda_min', default=-5, type=int, help='Minimum value of lambda: 10^x')
    parser.add_argument('--lambda_max', default=1, type=int, help='Maximum value of lambda: 10^x')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate of (S)GD: Currently has no effect since Adam is used')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size for algorithm 2')
    parser.add_argument('--n_epochs', default=500, type=int, help='Number of Epochs of (S)GD')
    parser.add_argument('--plot_convergence', default=False, action='store_true', help='If Convergence plot should be done')
    parser.add_argument('--dataset', help='Dataset to use', choices=['CommunitiesCrime', 'BarPass', 'StudentsMath', 'StudentsPortugese'])
    parser.add_argument('--nlambda', help='Number of lambda candidates', type=int, default=50)
    parser.add_argument('--weight_decay', help='SGD weight decay', type=float, default=0.0)
    args = parser.parse_args()
    run(args)