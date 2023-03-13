import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from utils import measures_from_Yhat

tau = 0.5

# Approximation of Q-function given by López-Benítez & Casadevall (2011) based on a second-order exponential function & Q(x) = 1- Q(-x):
a = 0.4920
b = 0.2887
c = 1.1893
Q_function = lambda x: torch.exp(-a*x**2 - b*x - c) 

def CDF_tau(Yhat, h=0.01, tau=0.5):
    m = len(Yhat)
    Y_tilde = (tau-Yhat)/h
    sum_ = torch.sum(Q_function(Y_tilde[Y_tilde>0])) \
           + torch.sum(1-Q_function(torch.abs(Y_tilde[Y_tilde<0]))) \
           + 0.5*(len(Y_tilde[Y_tilde==0]))
    return sum_/m

def Huber_loss(x, delta):
    if x.abs() < delta:
        return (x ** 2) / 2
    return delta * (x.abs() - delta / 2)

def train_fair_classifier(dataset, net, optimizer, lr_scheduler, fairness, lambda_, h, delta, device, n_epochs=200, batch_size=2048, seed=0):
    
    # Retrieve train/test splitted pytorch tensors for index=split
    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors
    
    # Retrieve train/test splitted numpy arrays for index=split
#     train_arrays, test_arrays = dataset.get_dataset_in_ndarray()
#     X_train_np, Y_train_np, Z_train_np, XZ_train_np = train_arrays
#     X_test_np, Y_test_np, Z_test_np, XZ_test_np = test_arrays

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    
    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi) #normal distribution
    
    # An empty dataframe for logging experimental results
    df = pd.DataFrame()
    df_ckpt = pd.DataFrame()
    
    loss_function = nn.BCELoss()
    costs = []
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
            xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = net(xz_batch)
            Ytilde = torch.round(Yhat.detach().reshape(-1))
            cost = 0
            dtheta = 0
            m = z_batch.shape[0]

            # prediction loss
            p_loss = loss_function(Yhat.squeeze(), y_batch)
            cost += (1 - lambda_) * p_loss

            # DP_Constraint
            if fairness == 'DP':
                Pr_Ytilde1 = CDF_tau(Yhat.detach(),h,tau)
                for z in range(1):
                    Pr_Ytilde1_Z = CDF_tau(Yhat.detach()[z_batch==z],h,tau)
                    m_z = z_batch[z_batch==z].shape[0]

                    Delta_z = Pr_Ytilde1_Z-Pr_Ytilde1
                    Delta_z_grad = torch.dot(phi((tau-Yhat.detach()[z_batch==z])/h).view(-1), 
                                              Yhat[z_batch==z].view(-1))/h/m_z
                    Delta_z_grad -= torch.dot(phi((tau-Yhat.detach())/h).view(-1), 
                                              Yhat.view(-1))/h/m

                    if Delta_z.abs() >= delta:
                        if Delta_z > 0:
                            Delta_z_grad *= lambda_*delta
                            cost += Delta_z_grad
                        else:
                            Delta_z_grad *= -lambda_*delta
                            cost += Delta_z_grad
                    else:
                        Delta_z_grad *= lambda_*Delta_z
                        cost += Delta_z_grad

            # EO_Constraint
            elif fairness == 'EO':
                for y in [0,1]:
                    Pr_Ytilde1_Y = CDF_tau(Yhat[y_batch==y].detach(),h,tau)
                    m_y = y_batch[y_batch==y].shape[0]
                    for z in range(1):
                        Pr_Ytilde1_ZY = CDF_tau(Yhat[(y_batch==y) & (z_batch==z)].detach(),h,tau)
                        m_zy = z_batch[(y_batch==y) & (z_batch==z)].shape[0]
                        Delta_zy = Pr_Ytilde1_ZY-Pr_Ytilde1_Y
                        Delta_zy_grad = torch.dot(
                                                  phi((tau-Yhat[(y_batch==y) & (z_batch==z)].detach())/h).view(-1), 
                                                  Yhat[(y_batch==y) & (z_batch==z)].view(-1)
                                                  )/h/m_zy
                        Delta_zy_grad -= torch.dot(
                                                   phi((tau-Yhat[y_batch==y].detach())/h).view(-1), 
                                                   Yhat[y_batch==y].view(-1)
                                                   )/h/m_y

                        if Delta_zy.abs() >= delta:
                            if Delta_zy > 0:
                                Delta_zy_grad *= lambda_*delta
                                cost += Delta_zy_grad
                            else:
                                Delta_zy_grad *= lambda_*delta
                                cost += -lambda_*delta*Delta_zy_grad
                        else:
                            Delta_zy_grad *= lambda_*Delta_zy
                            cost += Delta_zy_grad

            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
            costs.append(cost.item())
            
            # Print the cost per 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
                print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch+1, n_epochs,
                                                                          i+1, len(data_loader),
                                                                          cost.item()), end='\r')
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        Yhat_train = net(XZ_train).squeeze().detach().cpu().numpy()
        df_temp = measures_from_Yhat(Y_train_np, Z_train_np, Yhat=Yhat_train, threshold=tau)
        df_temp['epoch'] = epoch * len(data_loader) + i + 1
        df_ckpt = df_ckpt.append(df_temp)

        # Plot (cost, train accuracies, fairness measures) curves per 50 epochs
        if (epoch + 1) % 50 == 0:
            IPython.display.clear_output()
            print('Currently working on - seed: {}'.format(seed))
            plt.figure(figsize=(15,5), dpi=100)
            plt.subplot(1,3,1)
            plt.plot(costs)
            plt.xlabel('x10 iterations')
            plt.title('cost')
            plt.subplot(1,3,2)
            plt.plot(df_ckpt['acc'].to_numpy())
            plt.xlabel('epoch')
            plt.title('Accuracy')
            plt.subplot(1,3,3)
            if fairness == 'DP':
                plt.plot(df_ckpt['DDP'].to_numpy())
                plt.title('DDP')
            elif fairness == 'EO':
                plt.plot(df_ckpt['DEO'].to_numpy())
                plt.title('DEO')
            plt.xlabel('epoch')
            plt.show()
    
    Yhat_test = net(XZ_test).squeeze().detach().cpu().numpy()
    df_test = measures_from_Yhat(Y_test_np, Z_test_np, Yhat=Yhat_test, threshold=tau)
    
    return df_test