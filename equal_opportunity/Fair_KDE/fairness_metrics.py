import torch

import cvxpy as cp
import numpy as np

# +------------------------------------------+
# | Metric 1: Energy Distance                |
# +------------------------------------------+

def energy_distance(y1, y2):
    '''
    Compute energy distance between empirical distance y1 and y2, each 1 dimensional

    Args:
        y1 (torch.Tensor):  Samples from Distribution 1
        y2 (torch.Tensor):  Samples from Distribution 2

    Returns:
        dist (torch.Tensor): The computed Energy distance

    '''
    return (2*torch.abs(y1.unsqueeze(0)-y2.unsqueeze(1)).mean()
            -torch.abs(y1.unsqueeze(0)-y1.unsqueeze(1)).mean()
            -torch.abs(y2.unsqueeze(0)-y2.unsqueeze(1)).mean())

def energy_distance_forloop(y1, y2):
    '''
    Compute energy distance between empirical distance y1 and y2, each 1 dimensional

    Args:
        y1 (torch.Tensor):  Samples from Distribution 1
        y2 (torch.Tensor):  Samples from Distribution 2

    Returns:
        dist (torch.Tensor): The computed Energy distance

    '''
    d11 = torch.tensor(0.)
    d12 = torch.tensor(0.)
    d22 = torch.tensor(0.)

    for y_ in y1:
        d11 += (y_-y1).abs().mean()
        d12 += (y_-y2).abs().mean()
    d11 = d11/(y1.shape[0])
    d12 = d12/(y1.shape[0])
    for y_ in y2:
        d22 += (y_-y2).abs().mean()
    d22 = d22/(y2.shape[0])

    
    return 2*d12-d11-d22

# +------------------------------------------+
# | Metric 2: Wasserstein Distance       |
# +------------------------------------------+

def W1dist(y1,y2):
    '''
    Compute type 1 Wasserstein distance between empirical distribution y1 and y2, each 1 dimensional

    Args:
        y1 (torch.Tensor):  Samples from Distribution 1
        y2 (torch.Tensor):  Samples from Distribution 2

    Returns:
        dist (torch.Tensor): The computed Wasserstein distance
    '''
    # compute cost matrix
    C = torch.abs(y1.unsqueeze(0)-y2.unsqueeze(1))
    C_np = C.data.numpy()
    # solve OT problem
    T = cp.Variable(C_np.shape)
    ones_1 = np.ones((C_np.shape[0], 1))
    ones_2 = np.ones((C_np.shape[1], 1))
    objective = cp.Minimize(cp.sum(cp.multiply(C_np,T)))
    constraints = [
        T >=0,
        T@ones_2==ones_1/len(ones_1),
        T.T@ones_1==ones_2/len(ones_2)
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI)
    # objective value for gradient computation
    return (torch.Tensor(T.value)*C).sum()

# +------------------------------------------+
# | Evaluation Metric 1: Statistical Parity  |
# +------------------------------------------+

def statistical_parity(y1_hat, y2_hat, y1, y2):
    '''
    Compute max statistical imparity. This is equivalent to max
    difference in cdf

    Args:
        y1_hat (torch.Tensor):  Predictions for protected class 1
        y2_hat (torch.Tensor):  Predictions for protected class 2
        y1 (torch.Tensor):      True Value for protected class 1
        y2 (torch.Tensor):      True Value for protected class 2

    Returns:
        epsilon (torch.Tensor): max statistical imparity
    '''
    diff = torch.tensor(0)
    for y_test in torch.hstack((y1_hat,y2_hat)).flatten():
        cdf1_y = (y1_hat<=y_test).float().mean()
        cdf2_y = (y2_hat<=y_test).float().mean()
        if (cdf1_y-cdf2_y).abs()>diff:
            diff = (cdf1_y-cdf2_y).abs()
    return diff




# +------------------------------------------+
# | Evaluation Metric 2: Bounded Group Loss  |
# +------------------------------------------+

def bounded_group_loss(y1_hat, y2_hat, y1, y2, loss='L2'):
    '''
    Compute fraction in group loss between prediction for different 
    classes

    Args:
        y1_hat (torch.Tensor):  Predictions for protected class 1
        y2_hat (torch.Tensor):  Predictions for protected class 2
        y1 (torch.Tensor):      True Value for protected class 1
        y2 (torch.Tensor):      True Value for protected class 2

    Returns:
        epsilon (torch.Tensor): The difference between group loss
    '''
    r1 = y1_hat-y1
    r2 = y2_hat-y2
    if loss=='L2':
        lossf = lambda ra,rb: (ra**2).mean() / (rb**2).mean()
    if loss=='L1':
        lossf = lambda ra,rb: ra.abs().mean() / rb.abs().mean()
    l = lossf(r1,r2)
    return l if l<1 else 1/l

# +------------------------------------------+
# | Evaluation Metric 3:                     |
# | Group Fairness in Expectation            |
# +------------------------------------------+

def group_fair_expect(y1_hat, y2_hat, y1, y2):
    '''
    Compute Group Fairness in Expectation between prediction for different 
    classes

    Args:
        y1_hat (torch.Tensor):  Predictions for protected class 1
        y2_hat (torch.Tensor):  Predictions for protected class 2
        y1 (torch.Tensor):      True Value for protected class 1
        y2 (torch.Tensor):      True Value for protected class 2

    Returns:
        epsilon (torch.Tensor): The difference between means
    '''
    return (y1_hat.mean()-y2_hat.mean()).abs()

# +------------------------------------------+
# | Evaluation Metric 1: Statistical Parity  |
# +------------------------------------------+

def statistical_parity_classification(y1_hat, y2_hat, y1, y2):
    '''
    Compute max statistical imparity. This is equivalent to max
    difference in cdf

    Args:
        y1_hat (torch.Tensor):  Predictions for protected class 1
        y2_hat (torch.Tensor):  Predictions for protected class 2
        y1 (torch.Tensor):      True Value for protected class 1
        y2 (torch.Tensor):      True Value for protected class 2

    Returns:
        epsilon (torch.Tensor): max statistical imparity
    '''
    return ((y1_hat).sum() / y1_hat.shape[0] - (y2_hat).sum() / y2_hat.shape[0]).abs()


# +------------------------------------------+
# | Evaluation Metric 4: lp distance         |
# +------------------------------------------+ 

def lp_dist(y1_hat, y2_hat, y1, y2, p=1):
    '''
    Compute lp distance.

    Args:
        y1_hat (torch.Tensor):  Predictions for protected class 1
        y2_hat (torch.Tensor):  Predictions for protected class 2
        y1 (torch.Tensor):      True Value for protected class 1
        y2 (torch.Tensor):      True Value for protected class 2

    Returns:
        epsilon (torch.Tensor): lp distance
    '''
    dist = torch.tensor(0.)
    ys, idx = torch.hstack((y1_hat,y2_hat)).flatten().sort()
    for i in range(ys.shape[0]-1):
        cdf1_y = (y1_hat <= ys[i]).float().mean()
        cdf2_y = (y2_hat <= ys[i]).float().mean()
        dist += ((cdf1_y - cdf2_y).abs() ** p) * (ys[i+1] - ys[i])
    return dist**(1/p) 

# +------------------------------------------+
# | Regression Metric 1: MSE                 |
# +------------------------------------------+
def MSE(y1_hat, y2_hat, y1, y2):
    '''
    Compute lp distance.

    Args:
        y1_hat (torch.Tensor):  Predictions for protected class 1
        y2_hat (torch.Tensor):  Predictions for protected class 2
        y1 (torch.Tensor):      True Value for protected class 1
        y2 (torch.Tensor):      True Value for protected class 2

    Returns:
        MSE (torch.Tensor): mean squared error
    '''
    yhats = torch.hstack((y1_hat,y2_hat)).flatten()
    ys = torch.hstack((y1,y2)).flatten()
    return ((ys-yhats)**2).mean()

def MAE(y1_hat, y2_hat, y1, y2):
    '''
    Compute lp distance.

    Args:
        y1_hat (torch.Tensor):  Predictions for protected class 1
        y2_hat (torch.Tensor):  Predictions for protected class 2
        y1 (torch.Tensor):      True Value for protected class 1
        y2 (torch.Tensor):      True Value for protected class 2

    Returns:
        MAE (torch.Tensor): mean absolute error
    '''
    yhats = torch.hstack((y1_hat,y2_hat)).flatten()
    ys = torch.hstack((y1,y2)).flatten()
    return (ys-yhats).abs().mean()

def accuracy(y1_hat, y2_hat, y1, y2):
    ys = torch.hstack((y1,y2)).flatten()
    yhats = torch.hstack((y1_hat, y2_hat)).flatten()
    total = ys.size(0)
    correct = (yhats == ys).sum().item()

#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#         100 * correct / total))
    return torch.tensor(correct / total * 100)


