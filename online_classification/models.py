# models.py
# models for regression

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script provides models for MFL and Oneta et al.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class LinearRegression(nn.Module):
    def __init__(self, k):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(k, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

class NeuralNetwork(nn.Module):
    def __init__(self, k):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(k, 20, bias=True)
        self.linear2 = torch.nn.Linear(20, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        self.output = self.linear2(x)
        return self.output
    
"""Model of MFL"""
class NeuralNetworkClassification(nn.Module):
    def __init__(self, k):
        super(NeuralNetworkClassification, self).__init__()
        self.linear1 = torch.nn.Linear(k, 16, bias=True)
        self.linear2 = torch.nn.Linear(16, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        self.output = self.linear2(x)
        return self.output
    
"""Model of Oneta et al."""
class NeuralNetwork_MMD(nn.Module):
    def __init__(self, k):
        super(NeuralNetwork_MMD, self).__init__()
        self.linear1 = torch.nn.Linear(k, 16, bias=True)
        self.sigmoid_ = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 1, bias=True)
        
    def first_layer(self, x):
        return self.sigmoid_((self.linear1(x)))

    def forward(self, x):
        self.output = self.linear2(self.sigmoid_((self.linear1(x))))
        return self.output

# loss_functions: MAE and MSE
def MSE(y_pred, y):
    return ((y_pred - y) ** 2).mean()

def MAE(y_pred, y):
    return (y_pred - y).abs().mean()