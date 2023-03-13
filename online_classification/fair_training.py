# fair_training.py
# training methods for fair regression
import torch
from torch.autograd import Variable
import torch.optim as optim
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script provides implementation of stochastic gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
# +--------------------------------------------------+
# |  Algorithm: Stratified SGD - Classification      |
# +--------------------------------------------------+
def fair_learning(generator, predict, fair_loss, params, lambda_, psi=None, verbose = False, logdata=None, gamma_scheduler=None, lr_decay=1, lr=1e-3, logfairloss=None, **kwargs):
    '''
    Train model using Algorithm 2, which uses mini-batch SGD to train accuracy_loss + lambda_ * fairness_loss.

    Args:
        generator (generator):     Generator which yields (X,Y,A)
        predict (fct handle):      Prediction function handle, maps X-->Y_hat
        fair_loss (fct handle):    Fairness Loss function handle, maps Y_hat_prot, Y_hat_unprot-->L_fair
        params (list of params):   List of learnable parameters, such as returned by "nn.parameters()" or list of torch Variables
        lambda_ (numeric):         Hyperparameter controlling influence of L_fair
        verbose (bool, optional):  Verbosity 
        logdata (None or tuple of  X,Y,A, all torch.Tensor, optional): data for keeping track of training process
        gamma_scheduler (optional): Learning Rate Scheduler
        logfairloss (optional):     Fairness log function to use for logging instead of fairloss
    
    Returns:
        Trainig Loss over Training if logdata is provided, but changes params
    '''
    if logfairloss==None:
        logfairloss = fair_loss

    optimizer = optim.Adam(params, lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    batch_reg_loss = []
    batch_fair_loss = []
    for iterate, (X,Y,A) in enumerate(generator):
        if logdata:
            with torch.no_grad():
                y_hat_log = predict(logdata[0])
                batch_reg_loss.append(criterion(y_hat_log, logdata[1]))
                y_hat_1_log = y_hat_log[logdata[2]==1]
                y_hat_0_log = y_hat_log[logdata[2]==0]
                batch_fair_loss.append(logfairloss(y_hat_1_log, y_hat_0_log))
        optimizer.zero_grad()
        # predict
        y_hat = predict(X)
        # compute regression and fairness loss
        L = criterion(y_hat, Y)
        y_after_sig = torch.sigmoid(y_hat)
        y_after_sig = y_after_sig[:, None]
        y_hat_1 = y_hat[A==1]
        y_hat_0 = y_hat[A==0]
        L_fair = fair_loss(y_hat_1, y_hat_0)
        # overall loss
        loss = L + lambda_ * L_fair

        # logging
        if verbose:
            print('Iterate {}: L_reg={}, L_fair={}'.format(iterate, L.data.item(), L_fair.data.item()))

        # gradient comuptation and optimizer step
        loss.backward()
        optimizer.step()
    return batch_reg_loss, batch_fair_loss


# +--------------------------------------------------+
# |  Algorithm: Stratified SGD - Regression          |
# +--------------------------------------------------+
def fair_learning_regression(generator, predict, fair_loss, params, lambda_, psi=None, verbose = False, logdata=None, gamma_scheduler=None, lr_decay=1, lr=1e-3, logfairloss=None, **kwargs):
    '''
    Train model using Algorithm, which uses mini-batch SGD to train accuracy_loss + lambda_ * fairness_loss.

    Args:
        generator (generator):     Generator which yields (X,Y,A)
        predict (fct handle):      Prediction function handle, maps X-->Y_hat
        fair_loss (fct handle):    Fairness Loss function handle, maps Y_hat_prot, Y_hat_unprot-->L_fair
        params (list of params):   List of learnable parameters, such as returned by "nn.parameters()" or list of torch Variables
        lambda_ (numeric):         Hyperparameter controlling influence of L_fair
        verbose (bool, optional):  Verbosity 
        logdata (None or tuple of  X,Y,A, all torch.Tensor, optional): data for keeping track of training process
        gamma_scheduler (optional): Learning Rate Scheduler
        logfairloss (optional):     Fairness log function to use for logging instead of fairloss
    
    Returns:
        Trainig Loss over Training if logdata is provided, but changes params
    '''
    if logfairloss==None:
        logfairloss = fair_loss
    
    optimizer = optim.Adam(params, lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    criterion = torch.nn.MSELoss()
    batch_reg_loss = []
    batch_fair_loss = []
    for iterate, (X,Y,A) in enumerate(generator):
        if logdata:
            with torch.no_grad():
                y_hat_log = predict(logdata[0])
                batch_reg_loss.append(criterion(y_hat_log, logdata[1]))
                y_hat_1_log = y_hat_log[logdata[2]==1]
                y_hat_0_log = y_hat_log[logdata[2]==0]
                batch_fair_loss.append(logfairloss(y_hat_1_log, y_hat_0_log))
        optimizer.zero_grad()
        # predict
        y_hat = predict(X)
        # compute regression and fairness loss
        L = criterion(y_hat, Y)
        y_hat_1 = y_hat[A==1]
        y_hat_0 = y_hat[A==0]
        L_fair = fair_loss(y_hat_1, y_hat_0)
        # overall loss
        loss = L + lambda_ * L_fair

        # logging
        if verbose:
            print('Iterate {}: L_reg={}, L_fair={}'.format(iterate, L.data.item(), L_fair.data.item()))

        # gradient comuptation and optimizer step
        loss.backward()
        optimizer.step()
    return batch_reg_loss, batch_fair_loss