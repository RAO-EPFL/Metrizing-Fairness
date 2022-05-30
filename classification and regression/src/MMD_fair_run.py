import models
import fairness_metrics
import data_loader
import MMD_fair
import argparse
import pandas as pd
import numpy as np
import time
import pickle
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides an implementation of the paper in 
https://papers.nips.cc/paper/2020/file/af9c0e0c1dee63e5acad8b7ed1a5be96-Paper.pdf
An example usage: 
python .\MMD_fair_run.py --dataset {} --nlambda {} --seed {}
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
    Model = models.NeuralNetwork_MMD
    reg_loss = models.MSE
    fair_loss = fairness_metrics.sinkhorn_diver
    lambda_candidates = np.logspace(args.lambda_min, args.lambda_max, num=args.nlambda)
    train_test_split_fin = 0
    n_iterates = args.n_iterates
    lr = 1e-1
    lr_decay = 0.99
    if args.dataset == 'CommunitiesCrimeClassification':
        ds = data_loader.CommunitiesCrimeClassification(a_inside_x=args.a_inside_x)
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
    logfairloss = fair_loss
    

    if args.dataset != 'Adult':
        ds.split_test()

    k = ds.get_k() # Dimension

    # metrics to evaluate
    metrics = {
        'statistical_parity' : fairness_metrics.statistical_parity,
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

    # storage of results
    results_train = []
    results_test = []

    # run the test for various lambdas
     # extract the data
    X, Y, A = ds.get_data()
    for lambda_ in lambda_candidates:
        print('Training MMD-Sinkhorn method, for lambda_: {}/{}, seed:{}'.format(lambda_, args.nlambda, args.seed))

        model = Model(k)
        train_metrics, test_metrics = MMD_fair.mmd_fair_traintest(ds, 
                                                                 model,
                                                                 reg_loss, 
                                                                 fair_loss, 
                                                                 lr, 
                                                                 n_iterates, 
                                                                 lambda_, 
                                                                 metrics, 
                                                                 psi=None, plot_convergence=args.plot_convergence, logfairloss=logfairloss, lr_decay=lr_decay)
        train_metrics['lambda_'] = lambda_
        test_metrics['lambda_'] = lambda_
        results_train.append(train_metrics)
        results_test.append(test_metrics)
        # save the results
        df_train = pd.DataFrame(data=results_train)
        df_test = pd.DataFrame(data=results_test)

    if args.a_inside_x:
        df_train.to_csv('results/NN_MMD_sinkhorn/{}_{}_Sinkhorn_AinX_train_{}.csv'.format(args.dataset, args.model, args.seed))

        df_test.to_csv('results/NN_MMD_sinkhorn/{}_{}_Sinkhorn_AinX_test_{}.csv'.format(args.dataset, args.model, args.seed))
    else:
        print('here')
        df_train.to_csv('results/NN_MMD_sinkhorn/{}_{}_Sinkhorn_train_{}.csv'.format(args.dataset, \
            args.model, args.seed))

        df_test.to_csv('results/NN_MMD_sinkhorn/{}_{}_Sinkhorn_test_{}.csv'.format(args.dataset, \
            args.model, args.seed))
        
    PARAMS = {'dataset':args.dataset, 
               'lr':lr, 'iterates':n_iterates,
              'seed':args.seed, 
              'nlambda': args.nlambda, 
              'lambda_min':args.lambda_min,
              'lambda_max':args.lambda_max,
             'algorihtm':'Gradient-Descent', 
              'model_details':model.state_dict,
              'L':'MSE',
              'fair_loss':'Sinkhorn',
              'a_inside_x':args.a_inside_x
             }
    with open('results/NN_MMD_sinkhorn/{}_{}_Sinkhorn.pkl'.format(args.dataset, args.seed), 'wb') as f:
            pickle.dump({**PARAMS}, f, protocol=pickle.HIGHEST_PROTOCOL)
    #

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--seed', default=0, help='Randomness seed', type=int)
    parser.add_argument('--model', default='NN', choices=['linearregression', 'NN'], help='Regression Model')
    parser.add_argument('--lambda_min', default=-5, type=int, help='Minimum value of lambda: 10^x')
    parser.add_argument('--lambda_max', default=2, type=int, help='Maximum value of lambda: 10^x')
    parser.add_argument('--lr', default=1e-3, type=float, help='Gradient descent')
    parser.add_argument('--n_iterates', default=500, type=int, help='Number of Iterates of GD')
    parser.add_argument('--plot_convergence', default=False, action='store_true', help='If Convergence plot should be done')
    parser.add_argument('--dataset', help='Dataset to use', choices=['CommunitiesCrimeClassification','Compas', 'LawSchool', 'Adult', 'Credit', 'Drug'])
    parser.add_argument('--nlambda', help='Number of lambda candidates', type=int, default=50)
    parser.add_argument('--a_inside_x', default=False, type=str2bool, help='The sensitive feature is in X')
    args = parser.parse_args()
    run(args)