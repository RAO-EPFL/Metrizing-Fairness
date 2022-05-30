# Baseline 1: https://arxiv.org/pdf/1706.02409.pdf
import cvxpy as cp
import numpy as np
import argparse
import pandas as pd
import torch
from zafar_method import funcs_disp_mist
from zafar_method.utils import *
import fairness_metrics
import data_loader
from zafar_method import utils
import numpy as np
from tqdm import tqdm
import cvxpy as cp
from collections import namedtuple
from sklearn.metrics import log_loss
from zafar_method import loss_funcs as lf  # loss funcs that can be optimized subject to various constraints
import pickle
from copy import deepcopy
import os, sys
# from generate_synthetic_data import *
from zafar_method import utils as ut
from zafar_method import funcs_disp_mist as fdm
import time
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides implementation of http://proceedings.mlr.press/v54/zafar17a/zafar17a.pdf.
gamma parameter is the accuracy fairness tradeoff of the model. 
An example usage is python zafar_classification.py --dataset {} --seed {} --nlambda {}
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
    gamma_candidates = np.logspace(args.lambda_min, args.lambda_max, num=args.nlambda)
    if args.dataset == 'CommunitiesCrimeClassification':
        ds = data_loader.CommunitiesCrimeClassification(a_inside_x=0)
    if args.dataset == 'Compas':
        ds = data_loader.Compas(a_inside_x=0)
    if args.dataset == 'LawSchool':
        ds = data_loader.LawSchool(a_inside_x=0)
    if args.dataset == 'Credit':
        ds = data_loader.Credit(a_inside_x=0)
    if args.dataset == 'Adult':
        ds = data_loader.Adult(a_inside_x=0)
        train_test_split_fin = 1
    if args.dataset == 'Drug':
        ds = data_loader.Drug(a_inside_x=0)

    if args.dataset != 'Adult':
        ds.split_test()
    k = ds.get_k()

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

    X, Y, A = ds.get_data()
    X_test, Y_test, A_test = ds.get_test_data()
    x_train = X.cpu().detach().numpy()
    Y_train = Y.cpu().detach().numpy().flatten()
    a_train = A.cpu().detach().numpy().flatten()
    x_test = X_test.cpu().detach().numpy()
    y_test = Y_test.cpu().detach().numpy().flatten()
    a_test = A_test.cpu().detach().numpy().flatten()

    loss_function = "logreg"  # perform the experiments with logistic regression
    Y_test_ = y_test.copy()
    Y_train_ = Y_train.copy()

    Y_test_[y_test == 0] = -1
    Y_train_[Y_train_ == 0] = -1
    # run the test for various lambdas
    y_train = Y_train_
    y_test = Y_test_
    x_control_train = {"s1": a_train}
    x_control_test =  {"s1": a_test}
    cons_params = None  # constraint parameters, will use them later
    EPS = 1e-6
    for gamma in gamma_candidates:
        print('Training Zafar method, for gamma: {}/{}, seed:{}'.format(gamma, args.nlambda, args.seed))
        start_time = time.time()
#         mult_range = np.arange(1.0, 0.0 - it, -it).tolist()
#         sensitive_attrs_to_cov_thresh = deepcopy(cov_all_train_uncons)
        apply_fairness_constraints = 0 # set this flag to one since we want to optimize accuracy subject to fairness constraints
        apply_accuracy_constraint = 1
        sep_constraint = 0

#         for m in mult_range:
#             sensitive_attrs_to_cov_thresh = deepcopy(cov_all_train_uncons)
#             for s_attr in sensitive_attrs_to_cov_thresh.keys():
#                 for cov_type in sensitive_attrs_to_cov_thresh[s_attr].keys():
#                     for s_val in sensitive_attrs_to_cov_thresh[s_attr][cov_type]:
#                         sensitive_attrs_to_cov_thresh[s_attr][cov_type][s_val] *= m
        sensitive_attrs_to_cov_thresh = {"s1":0}
       
        w = train_model(x_train, y_train, x_control_train, lf._logistic_loss,  apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, ['s1'], sensitive_attrs_to_cov_thresh, gamma)
#         y_test_predicted = np.sign(np.dot(x_test, w))
#         correct_answers = (y_test_predicted == y_test).astype(int) # will have 1 when the prediction and the actual label match
#         accuracy = float(sum(correct_answers)) / float(len(correct_answers))

#         y_test_predict[y_test_predict == -1] = 0
#         w = torch.tensor(w).float()
#         theta0 = torch.tensor(w).float()
        stop_time = time.time()
        predict = lambda X: torch.tensor(np.maximum(np.sign(np.dot(X.cpu().detach().numpy(), w)), 0)).float()
        
        # metrics on train set
        y_hat = predict(X).flatten()
        y_hat = y_hat.unsqueeze(1)
        y_hat_1 = y_hat[A==1]
        y_hat_0 = y_hat[A==0]
        y_1 = Y[A==1]
        y_0 = Y[A==0]
        train_results = {}
        for key in metrics.keys():
            train_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

        # metrics on test set
        y_hat = predict(X_test).flatten()
        y_hat = y_hat.unsqueeze(1)
        y_hat_1 = y_hat[A_test==1]
        y_hat_0 = y_hat[A_test==0]
        y_1 = Y_test[A_test==1]
        y_0 = Y_test[A_test==0]
        test_results = {}
        for key in metrics.keys():
            test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

        train_results['lambda_'] = gamma
        train_results['time'] = stop_time - start_time
        test_results['lambda_'] = gamma
        results_train.append(train_results)
        results_test.append(test_results)

    df_train = pd.DataFrame(data=results_train)
    df_test = pd.DataFrame(data=results_test)

    df_train.to_csv('results/zafar/{}_zafar_{}_train.csv'.format(args.dataset, args.seed))
    
    df_test.to_csv('results/zafar/{}_zafar_{}_test.csv'.format(args.dataset, args.seed))
            
    PARAMS = {'dataset':args.dataset, 
              'method':'zafar',
              'seed':args.seed, 
              'nlambda': args.nlambda, 
              'lambda_min':args.lambda_min,
              'lambda_max':args.lambda_max,
              'a_inside_x': False
             }
    with open('results/zafar/{}_zafar_{}.pkl'.format(args.dataset, args.seed), 'wb') as f:
            pickle.dump({**PARAMS}, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--seed', default=0, help='Randomness seed', type=int)
    parser.add_argument('--lambda_min', default=-5, type=int, help='Minimum value of lambda: 10^x')
    parser.add_argument('--lambda_max', default=1, type=int, help='Maximum value of lambda: 10^x')
    parser.add_argument('--dataset', help='Dataset to use', choices=['CommunitiesCrimeClassification', 'Compas', 'LawSchool', 'Adult', 'Credit', 'Drug'])
    parser.add_argument('--nlambda', help='Number of lambda candidates', type=int, default=25)
    args = parser.parse_args()
    run(args)