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
An example usage python run_benchmark.py --dataset {} --seed {} --a_inside_x True --nlambda {}
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
    Model = models.LinearRegression if args.model=='linear' else models.NeuralNetworkClassification
    fair_loss = fairness_metrics.energy_distance
    lambda_candidates = np.logspace(args.lambda_min, args.lambda_max, num=args.nlambda)
    train_test_split_fin = 0
    lr = args.lr
    n_epochs = args.n_epochs
    #lr = 5e-4
    n_epochs = 500
    lr_decay = 0.99
    batch_size = 2048
    if args.dataset == 'CommunitiesCrimeClassification':
        ds = data_loader.CommunitiesCrimeClassification(a_inside_x=args.a_inside_x)
        batch_size = 512
    if args.dataset == 'Compas':
        ds = data_loader.Compas(a_inside_x=args.a_inside_x)
    if args.dataset == 'LawSchool':
        ds = data_loader.LawSchool(a_inside_x=args.a_inside_x)
    if args.dataset == 'Credit':
        ds = data_loader.Credit(a_inside_x=args.a_inside_x)
    if args.dataset == 'Adult':
        ds = data_loader.Adult(a_inside_x=args.a_inside_x)
        train_test_split_fin = 1
    if args.dataset == 'Drug':
        ds = data_loader.Drug(a_inside_x=args.a_inside_x)
        batch_size = 512

    logfairloss = fair_loss
    if args.dataset != 'Adult':
        ds.split_test()

    k = ds.get_k() # Dimension

    # metrics to evaluate
    # CHANGE: CHANGE TO EQUAL OPPORTUNITY FORMULATION
    metrics = {
        'statistical_parity' : fairness_metrics.statistical_parity,
        'equal_opportunity' : fairness_metrics.equal_opportunity_classification,
        'statistical_parity_classification' : fairness_metrics.statistical_parity_classification,
        'bounded_group_loss_L1' : lambda y1_hat, y2_hat, y1, y2: fairness_metrics.bounded_group_loss(y1_hat, y2_hat, y1, y2, loss='L1'),
        'bounded_group_loss_L2' : fairness_metrics.bounded_group_loss,
        'group_fair_expect' : fairness_metrics.group_fair_expect,
        'l1_dist' : lambda y1_hat, y2_hat, y1, y2: fairness_metrics.lp_dist(y1_hat, y2_hat, y1, y2, p=1),
        'l2_dist' : lambda y1_hat, y2_hat, y1, y2: fairness_metrics.lp_dist(y1_hat, y2_hat, y1, y2, p=2),
        'MSE' : fairness_metrics.MSE,
        'MAE' : fairness_metrics.MAE,
        'accuracy' : fairness_metrics.accuracy
    }
    # END CHANGE

    # storage of results
    results_train = []
    results_test = []

    # run the test for various lambdas
    for lambda_ in lambda_candidates:
        print('Training Our method, for lambda_: {}/{}, seed:{}'.format(lambda_, args.nlambda, args.seed))

        model = Model(k)
        print(Model)
        train_metrics, test_metrics = benchmark.train_test_fair_learning(ds=ds, 
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
    if args.a_inside_x:
        df_train.to_csv('results/NN_energy/{}_{}_AinX_train_{}.csv'.format(args.dataset, \
            args.model, args.seed))

        df_test.to_csv('results/NN_energy/{}_{}_AinX_test_{}.csv'.format(args.dataset, \
            args.model, args.seed))
    else:
        print('here')
        df_train.to_csv('results/NN_energy/{}_{}_train_{}.csv'.format(args.dataset, \
            args.model, args.seed))

        df_test.to_csv('results/NN_energy/{}_{}_test_{}.csv'.format(args.dataset, \
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
              'L':'BCE_cross_entropy',
              'fair_loss':'Energy',
              'lr_decay':lr_decay,
              'a_inside_x':args.a_inside_x
             }
    with open('results/NN_energy/{}_{}_{}.pkl'.format(args.dataset, args.model, args.seed), 'wb') as f:
            pickle.dump({**PARAMS}, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--seed', default=0, help='Randomness seed', type=int)
    parser.add_argument('--model', default='NN', choices=['linear', 'NN'], help='Model')
#     parser.add_argument('--regloss', default='L2', choices=['L1', 'L2'], help='Regression Loss')
#     parser.add_argument('--fairloss', required=True, choices=['Energy', 'Wasserstein'], help='Fairness loss')
    parser.add_argument('--lambda_min', default=-5, type=int, help='Minimum value of lambda: 10^x')
    parser.add_argument('--lambda_max', default=2, type=int, help='Maximum value of lambda: 10^x')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning Rate of (S)GD: Currently has no effect since Adam is used')
    parser.add_argument('--n_epochs', default=500, type=int, help='Number of Epochs of (S)GD')
    parser.add_argument('--plot_convergence', default=False, action='store_true', help='If Convergence plot should be done')
    parser.add_argument('--dataset', help='Dataset to use', choices=['Synthetic1', 'Synthetic2', 'CommunitiesCrime', 'CommunitiesCrimeClassification',
                                                                     'BarPass', 'StudentsMath', 'StudentsPortugese', 'Compas', 'LawSchool', 'Adult', 
                                                                     'Credit', 'Drug'])
    parser.add_argument('--nlambda', help='Number of lambda candidates', type=int, default=50)
    parser.add_argument('--weight_decay', help='SGD weight decay', type=float, default=0.0)
    parser.add_argument('--a_inside_x', default=False, type=str2bool, help='The sensitive feature is in X')
    args = parser.parse_args()
    run(args)