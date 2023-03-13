# benchmark.py
# file with functions for running experiment
import fair_training
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time

def convergence_plotter(regloss, fairloss, lambda_):
    plt.figure(figsize=(16,5))
    plt.subplot(131)
    plt.plot(regloss)
    plt.title('Regression Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Regression Loss')

    plt.subplot(132)
    plt.plot(fairloss)
    plt.title('Fairness Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Fairness Loss')

    plt.subplot(133)
    plt.plot(lambda_*np.array(fairloss)+np.array(regloss))
    plt.title('Overall Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script provides implementaion train and test function for MFL.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def train_test_fair_learning(ds, model, fair_loss, lr, batch_size, N_epochs, lambda_, metrics, lr_decay = 1, psi=None, plot_convergence=False, logfairloss=None, train_test_split_fin=0, **kwargs):
    '''
    Train a model using algorithm 2 and test it on metrics

    Args:
        ds (data_loader.DataLoader):       Data loader to use
        model (torch.nn.Module):           Pytorch module
        fair_loss (fct handle):            Fairness Loss function handle, maps Y_hat_prot, Y_hat_unprot-->L_fair
        lr (float):                        SGD Learning Rate
        batch_size (int):                  Batch-Size of SGD
        N_epochs (int):                    Number of epochs for SGD
        lambda_ (numeric):                 Hyperparameter controlling influence of L_fair
        metrics (dict with fctn handles):  Metrics to use in evaluation. Will return a dict with same keys
        plot_convergence (bool, optional): If convergence plot of training should be shown
        logfairloss (fctn  handle, optional): Fairness function used for logging instead of fair_loss
        train_test_split_fin (bool, adult data) : Train-Test split already performed on adultdata                
    Returns:
        train_results, test_results: dicts of results
    '''
    # train the model
    start_time = time.time()
    regloss, fairloss = fair_training.fair_learning(generator=ds.stratified_batch_generator_worep(batch_size, N_epochs), 
                                                  predict=model.forward, 
                                                  fair_loss=fair_loss, 
                                                  params=model.parameters(),
                                                  lambda_=lambda_,
                                                  lr_decay=lr_decay,
                                                  logdata = ds.get_log_data() if plot_convergence else None,
                                                  psi=psi, lr=lr, logfairloss=logfairloss, **kwargs)
    stop_time = time.time()
    # plot convergence if desired
    if plot_convergence:
        convergence_plotter(regloss, fairloss, lambda_)
    # compute metrics
    model.eval()

    # metrics on training set
    if train_test_split_fin:
        X, Y, A, X_test, Y_test, A_test = ds.get_adult_data()
    else:   
        X, Y, A = ds.get_log_data()
        X_test, Y_test, A_test= ds.get_test_data()
        
    y_hat = torch.round(torch.sigmoid(model(X)))
    y_hat_1 = y_hat[A==1]
    y_hat_0 = y_hat[A==0]
    y_1 = Y[A==1]
    y_0 = Y[A==0]
    train_results = {}
    for key in metrics.keys():
        train_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

    # metrics on test set
    
    y_hat = torch.round(torch.sigmoid(model(X_test)))
    y_hat_1 = y_hat[A_test==1]
    y_hat_0 = y_hat[A_test==0]
    y_1 = Y_test[A_test==1]
    y_0 = Y_test[A_test==0]
    
    test_results = {}
    
    for key in metrics.keys():
        test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()
        
    train_results['time'] = stop_time - start_time

    return train_results, test_results

def train_test_fair_learning_regression(ds, model, fair_loss, lr, batch_size, N_epochs, lambda_, metrics, lr_decay = 1, psi=None, plot_convergence=False, logfairloss=None, train_test_split_fin=0, **kwargs):
    '''
    Train a model using algorithm 2 and test it on metrics

    Args:
        ds (data_loader.DataLoader):       Data loader to use
        model (torch.nn.Module):           Pytorch module
        fair_loss (fct handle):            Fairness Loss function handle, maps Y_hat_prot, Y_hat_unprot-->L_fair
        lr (float):                        SGD Learning Rate
        batch_size (int):                  Batch-Size of SGD
        N_epochs (int):                    Number of epochs for SGD
        lambda_ (numeric):                 Hyperparameter controlling influence of L_fair
        metrics (dict with fctn handles):  Metrics to use in evaluation. Will return a dict with same keys
        plot_convergence (bool, optional): If convergence plot of training should be shown
        logfairloss (fctn  handle, optional): Fairness function used for logging instead of fair_loss
        train_test_split_fin (bool, adult data) : Train-Test split already performed on adultdata                
    Returns:
        train_results, test_results: dicts of results
    '''
    raise NotImplementedError
    # train the model
    start_time = time.time()
    regloss, fairloss = fair_training.fair_learning_regression(generator=ds.stratified_batch_generator_worep(batch_size, N_epochs), 
                                                  predict=model.forward, 
                                                  fair_loss=fair_loss, 
                                                  params=model.parameters(),
                                                  lambda_=lambda_,
                                                  lr_decay=lr_decay,
                                                  logdata = ds.get_log_data() if plot_convergence else None,
                                                  psi=psi, lr=lr, logfairloss=logfairloss, **kwargs)
    stop_time = time.time()
    # plot convergence if desired
    if plot_convergence:
        convergence_plotter(regloss, fairloss, lambda_)
    # compute metrics
    model.eval()

    # metrics on training set
    X, Y, A = ds.get_log_data()
    X_test, Y_test, A_test = ds.get_test_data()
        
    y_hat = model(X)
    y_hat_1 = y_hat[A==1]
    y_hat_0 = y_hat[A==0]
    y_1 = Y[A==1]
    y_0 = Y[A==0]
    train_results = {}
    for key in metrics.keys():
        train_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()

    # metrics on test set
    
    y_hat = model(X_test)
    y_hat_1 = y_hat[A_test==1]
    y_hat_0 = y_hat[A_test==0]
    y_1 = Y_test[A_test==1]
    y_0 = Y_test[A_test==0]
    
    test_results = {}
    
    for key in metrics.keys():
        test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()
        
    train_results['time'] = stop_time - start_time

    return train_results, test_results

