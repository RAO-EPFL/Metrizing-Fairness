# fair_training.py
# training methods for fair regression
import torch
from torch.autograd import Variable
import torch.optim as optim
import time
from tqdm import tqdm
# +---------------------------------+
# |  Algorithm 1: Gradient Descent  |
# +---------------------------------+
"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides an implementation of the paper in 
https://papers.nips.cc/paper/2020/file/af9c0e0c1dee63e5acad8b7ed1a5be96-Paper.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def mmd_gradient_descent(X, Y, A, model, predict, reg_loss, fair_loss, params, lr, N_iterates, lambda_, verbose=False, log=False, logfairloss=None, lr_decay=1, **kwargs):
    '''
    Train model using Algorithm 1, which uses simple gradient descent.

    Args:
        X (torch.Tensor):          X data
        Y (torch.Tensor):          Y data
        A (torch.Tensor):          A data
        predict (fct handle):      Prediction function handle, maps X-->Y_hat
        reg_loss (fct handle):     Regression Loss function handle, maps Y_hat, Y-->L_reg
        fair_loss (fct handle):    Fairness Loss function handle, maps Y_hat_prot, Y_hat_unprot-->L_fair
        params (list of params):   List of learnable parameters, such as returned by "nn.parameters()" or list of torch Variables
        lr (float):                SGD Learning Rate
        N_iterates (int):          Number of iterates for SGD
        lambda_ (numeric):         Hyperparameter controlling influence of L_fair
        psi (fct handle, optional):Transformation function maps from Y_hat, Y --> score, fair_loss is computed on score
        verbose (bool, optional):  Verbosity 
        log (bool, optional):      Return training path
        logfairloss (optional):     Sinkhorn divergence
    
    Returns:
        Trainig Loss over Training if log=True, but changes params
    '''
    optimizer = optim.SGD(params, lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    criterion = torch.nn.BCEWithLogitsLoss()


#     optimizer = optim.Adam(params)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    epoch_reg_loss = []
    epoch_fair_loss = []
    for iterate in tqdm(range(N_iterates)):
        # zero grad accumulator
        optimizer.zero_grad()
        # predict
        y_hat = predict(X)
        y_hat_first_layer = model.first_layer(X)
        L_reg = criterion(y_hat, Y)
#         y_hat = torch.sigmoid(y_hat)
        # compute regression and fairness loss
        y_hat_1 = y_hat_first_layer[(A.squeeze()==1) & (Y.squeeze()==1)]
        y_hat_0 = y_hat_first_layer[(A.squeeze()==0) & (Y.squeeze()==1)]
        L_fair = fair_loss(y_hat_1, y_hat_0)
        
#         all_linear1_params = torch.cat([x.view(-1) for x in model.linear1.parameters()])
#         all_linear2_params = torch.cat([x.view(-1) for x in model.linear2.parameters()])
#         W_froben = torch.norm(all_linear1_params, 2) ** 2 
#         V_froben = torch.norm(all_linear2_params, 2) ** 2

        # overall loss
#         reg_weight = 0.1
        loss = L_reg + lambda_ * L_fair 
        
        # logging
        if verbose:
            print('Iterate {}: L_reg={}, L_fair={}'.format(iterate, L_reg.data.item(), L_fair.data.item()))
        if log:
            epoch_fair_loss.append(L_fair.data.item())
            epoch_reg_loss.append(L_reg.data.item())
        
        # gradient comuptation and optimizer step
        loss.backward()
        optimizer.step()
        #scheduler.step()
    return  epoch_reg_loss, epoch_fair_loss



def mmd_fair_traintest(ds, model, reg_loss, fair_loss, lr, n_iterates, lambda_, metrics, psi=None, plot_convergence=False, logfairloss=None, train_test_split_fin=0, lr_decay=1, **kwargs):
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
    if train_test_split_fin:
        X, Y, A, X_test, Y_test, A_test = ds.get_adult_data()
    else:   
        X, Y, A = ds.get_log_data()
        X_test, Y_test, A_test= ds.get_test_data()
    regloss, fairloss = mmd_gradient_descent(X, Y, A, model, model.forward, 
                                                  reg_loss,
                                                  fair_loss, 
                                                  model.parameters(), lr, n_iterates,
                                                  lambda_,
                                                  logdata = ds.get_log_data() if plot_convergence else None, logfairloss=logfairloss, lr_decay=lr_decay, **kwargs)
    # plot convergence if desired
    if plot_convergence:
        convergence_plotter(regloss, fairloss, lambda_)
    # compute metrics
    model.eval()

    # metrics on training seed
    stop_time = time.time()
    y_hat = torch.round(torch.sigmoid(model.forward(X)))
    y_hat_1 = y_hat[A==1]
    y_hat_0 = y_hat[A==0]
    y_1 = Y[A==1]
    y_0 = Y[A==0]
    train_results = {}
    for key in metrics.keys():
        train_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()
    train_results['time'] = stop_time - start_time
    # metrics on test set
    
    y_hat = torch.round(torch.sigmoid(model.forward(X_test)))
    y_hat_1 = y_hat[A_test==1]
    y_hat_0 = y_hat[A_test==0]
    y_1 = Y_test[A_test==1]
    y_0 = Y_test[A_test==0]
    test_results = {}
    for key in metrics.keys():
        test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()
    return train_results, test_results
        