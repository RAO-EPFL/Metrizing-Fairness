import cvxpy as cp
import numpy as np
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import fairness_metrics
import data_loader
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides an implementation of the paper in 
https://arxiv.org/pdf/1706.02409.pdf
An example usage: 
python .\baseline_convex_fair_regression.py --seed {} --fairness {} --dataset {}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def run(args):
    # act on experiment parameters:
    data_loader.set_seed(args.seed)
    lambda_candidates = np.logspace(args.lambda_min, args.lambda_max, num=args.nlambda)
    if args.dataset == 'CommunitiesCrime':
        ds = data_loader.CommunitiesCrime()
    if args.dataset == 'BarPass':
        ds = data_loader.BarPass()
    if args.dataset == 'StudentsMath':
        ds = data_loader.StudentPerformance(subject='Math')
    if args.dataset == 'StudentsPortugese':
        ds = data_loader.StudentPerformance(subject='Portugese')

    ds.split_test()
    k = ds.get_k()

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

    # get data
    X0, Y0 = ds.get_data_for_A(0)
    X0 = X0.numpy()
    Y0 = Y0.numpy()
    X1, Y1 = ds.get_data_for_A(1)
    X1 = X1.numpy()
    Y1 = Y1.numpy()

    X, Y, A = ds.get_data()
    X_test, Y_test, A_test = ds.get_test_data()

    # run the test for various lambdas
    for lambda_ in lambda_candidates:
        start_time = time.time()
        if args.fairness == 'group':
            D = np.exp(-(Y1-Y0.T)**2)
            n1n0 = D.shape[0]*D.shape[1]
            theta = cp.Variable((X1.shape[1],1))
            theta0 = cp.Variable()

            objective = cp.Minimize(cp.sum((Y0-theta0-X0@theta)**2)/Y0.shape[0]+\
                                    cp.sum((Y1-theta0-X1@theta)**2)/Y1.shape[0]+\
                                    lambda_*(cp.sum(cp.multiply(D, (X1@theta - (X0@theta).T)))/n1n0)**2 +\
                                    args.gamma*(theta0**2 + cp.sum_squares(theta)))
            problem = cp.Problem(objective, [])
            problem.solve(solver = cp.GUROBI, verbose=False)
        else:
            D = np.exp(-(Y1-Y0.T)**2)
            n1n0 = D.shape[0]*D.shape[1]
            theta = cp.Variable((X1.shape[1],1))
            theta0 = cp.Variable()

            objective = cp.Minimize(cp.sum((Y0-theta0-X0@theta)**2)/Y0.shape[0]+\
                                    cp.sum((Y1-theta0-X1@theta)**2)/Y1.shape[0]+\
                                    lambda_*(cp.sum(cp.multiply(D, (X1@theta - (X0@theta).T)**2))/n1n0) +\
                                    args.gamma*(theta0**2 + cp.sum_squares(theta)))
            problem = cp.Problem(objective, [])
            problem.solve(solver = cp.GUROBI, verbose=False)
        duration = time.time()-start_time
        
        theta = torch.tensor(theta.value).float()
        theta0 = torch.tensor(theta0.value).float()
        predict = lambda X: theta0 + X@theta

        # metrics on train set
        y_hat = predict(X)
        y_hat_1 = y_hat[A==1]
        y_hat_0 = y_hat[A==0]
        y_1 = Y[A==1]
        y_0 = Y[A==0]
        train_results = {}
        for key in metrics.keys():
            train_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

        # metrics on test set
        y_hat = predict(X_test)
        y_hat_1 = y_hat[A_test==1]
        y_hat_0 = y_hat[A_test==0]
        y_1 = Y_test[A_test==1]
        y_0 = Y_test[A_test==0]
        test_results = {}
        for key in metrics.keys():
            test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

        train_results['lambda_'] = lambda_
        test_results['lambda_'] = lambda_
        train_results['time'] = duration
        test_results['time'] = duration
        results_train.append(train_results)
        results_test.append(test_results)

    df_train = pd.DataFrame(data=results_train)
    df_test = pd.DataFrame(data=results_test)

    df_train.to_csv('results_regression/cvx_regression_baseline/{}_cvx-bl-{}_{}_train_{}.csv'.format(args.dataset, \
        args.gamma, \
        args.fairness, args.seed))
    
    df_test.to_csv('results_regression/cvx_regression_baseline/{}_cvx-bl-{}_{}_test_{}.csv'.format(args.dataset, \
        args.gamma, \
        args.fairness, args.seed))
            
def run_sgd(args):
    # act on experiment parameters:
    data_loader.set_seed(args.seed)
    lambda_candidates = np.logspace(args.lambda_min, args.lambda_max, num=args.nlambda)
    if args.dataset == 'CommunitiesCrime':
        ds = data_loader.CommunitiesCrime()
    if args.dataset == 'BarPass':
        ds = data_loader.BarPass()
    if args.dataset == 'StudentsMath':
        ds = data_loader.StudentPerformance(subject='Math')
    if args.dataset == 'StudentsPortugese':
        ds = data_loader.StudentPerformance(subject='Portugese')

    ds.split_test()
    k = ds.get_k()

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

    # get data
    X0, Y0 = ds.get_data_for_A(0)
    X1, Y1 = ds.get_data_for_A(1)

    X, Y, A = ds.get_data()
    k = X.shape[1]
    X_test, Y_test, A_test = ds.get_test_data()

    # run the test for various lambdas
    for lambda_ in lambda_candidates:
        D = torch.exp(-(Y1-Y0.T)**2)
        objective_group = lambda theta0, theta: (torch.sum((Y0-theta0-X0@theta)**2)+torch.sum((Y1-theta0-X1@theta)**2))/(D.shape[0]+D.shape[1])+\
                                lambda_*(torch.mean(D*(X1@theta - (X0@theta).T)))**2

        objective_individual = lambda theta0, theta: (torch.sum((Y0-theta0-X0@theta)**2)+\
                                                torch.sum((Y1-theta0-X1@theta)**2))/(D.shape[0]+D.shape[1])+\
                                                lambda_*(torch.mean(D*(X1@theta - (X0@theta).T)**2))

        objective = objective_group if args.fairness=='group' else objective_individual
        theta_0 = torch.rand(1)
        theta = torch.rand([k, 1])
        theta_0.requires_grad = True
        theta.requires_grad = True
        optimizer = torch.optim.Adam([theta_0, theta])
        losses = []
        for epoch in tqdm(range(5000)):
            optimizer.zero_grad()
            loss = objective(theta_0, theta)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        predict = lambda X: theta_0 + X@theta

        # metrics on train set
        y_hat = predict(X)
        y_hat_1 = y_hat[A==1]
        y_hat_0 = y_hat[A==0]
        y_1 = Y[A==1]
        y_0 = Y[A==0]
        train_results = {}
        for key in metrics.keys():
            train_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

        # metrics on test set
        y_hat = predict(X_test)
        y_hat_1 = y_hat[A_test==1]
        y_hat_0 = y_hat[A_test==0]
        y_1 = Y_test[A_test==1]
        y_0 = Y_test[A_test==0]
        test_results = {}
        for key in metrics.keys():
            test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

        train_results['lambda_'] = lambda_
        test_results['lambda_'] = lambda_
        results_train.append(train_results)
        results_test.append(test_results)

    df_train = pd.DataFrame(data=results_train)
    df_test = pd.DataFrame(data=results_test)

    df_train.to_csv('results/cvx_regression_baseline/{}_cvx-bl-{}_{}_train_{}.csv'.format(args.dataset, \
        args.gamma, \
        args.fairness, args.seed))
    
    df_test.to_csv('results/cvx_regression_baseline/{}_cvx-bl-{}_{}_test_{}.csv'.format(args.dataset, \
        args.gamma, \
        args.fairness, args.seed))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--seed', default=0, help='Randomness seed', type=int)
    parser.add_argument('--lambda_min', default=-2, type=int, help='Minimum value of lambda: 10^x')
    parser.add_argument('--lambda_max', default=5, type=int, help='Maximum value of lambda: 10^x')
    parser.add_argument('--dataset', help='Dataset to use', choices=['CommunitiesCrime', 'BarPass', 'StudentsMath', 'StudentsPortugese'])
    parser.add_argument('--nlambda', help='Number of lambda candidates', type=int, default=50)
    parser.add_argument('--gamma', help='Weight of L2 regularizer', type=float, default=0)
    parser.add_argument('--fairness', help='Fairness Type to use', choices=['group', 'individual'])
    args = parser.parse_args()
    run(args)