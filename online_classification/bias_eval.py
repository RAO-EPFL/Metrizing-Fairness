import torch
import data_loader
import models
import fairness_metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def find_batchsize(N_target, A):
    candidate_1 = torch.argmax((A.flatten().cumsum(0)==2).int()).item() + 1
    candidate_0 = torch.argmax(((1-A).flatten().cumsum(0)==2).int()).item() + 1
    if candidate_0<2 or candidate_1<0:
        return -1
    return max(N_target, candidate_1, candidate_0)

def accuracy_1(y_hat_1, y_hat_0, y_1, y_0):
    return torch.Tensor((y_hat_1==y_1).float().mean())
def accuracy_0(y_hat_1, y_hat_0, y_1, y_0):
    return torch.Tensor((y_hat_0==y_0).float().mean())

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
    'accuracy' : fairness_metrics.accuracy,
    'accuracy1' : accuracy_1,
    'accuracy0' : accuracy_0
}

def train_metrics_fullbias(target_batchsize, seed=0, plot=False, lambda_=1):
    lr = 5e-4
    drug = data_loader.Drug(True)
    model = models.NeuralNetworkClassification(drug.get_k())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    fairloss = fairness_metrics.energy_distance_biased

    data_loader.set_seed(seed)
    drug.split_test()
    X, Y, A = drug.get_data()
    X_test, Y_test, A_test = drug.get_test_data()
    
    N_epochs = 500

    losses = []
    test_losses = []
    for epoch in range(N_epochs):
        X,Y,A = shuffle(X,Y,A)
        sumloss = 0
        batchstart = 0
        while batchstart<len(A):
            optimizer.zero_grad()
            batchsize = find_batchsize(target_batchsize, A[batchstart:])
            if batchsize>0:
                X_batch, Y_batch, A_batch = X[batchstart:batchstart+batchsize], Y[batchstart:batchstart+batchsize], A[batchstart:batchstart+batchsize]
                batchstart = batchstart+batchsize
                pred = model(X_batch)
                L = criterion(pred, Y_batch)
                y_after_sig = torch.sigmoid(pred)
                y_after_sig = y_after_sig[:, None]
                y_hat_1 = pred[A_batch.flatten()==1]
                y_hat_0 = pred[A_batch.flatten()==0]
                L_fair = fairloss(y_hat_1, y_hat_0)
                # overall loss
                loss = L + lambda_ * L_fair
                loss.backward()
                sumloss += loss.detach().item()
                optimizer.step()
            else:
                batchstart = len(A)+1
        losses.append(sumloss)
        with torch.no_grad():
            pred = model(X_test)
            y_hat_1 = pred[A_test.flatten()==1]
            y_hat_0 = pred[A_test.flatten()==0]
            testloss = criterion(model(X_test), Y_test) + fairloss(y_hat_1, y_hat_0)
            test_losses.append(testloss)

    if plot:
        plt.plot(losses)

    y_hat = torch.round(torch.sigmoid(model(X_test)))
    y_hat_1 = y_hat[A_test.flatten()==1].flatten()
    y_hat_0 = y_hat[A_test.flatten()==0].flatten()
    y_1 = Y_test[A_test.flatten()==1].flatten()
    y_0 = Y_test[A_test.flatten()==0].flatten()

    test_results = {}

    for key in metrics.keys():
        test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()
    return test_results, losses, test_losses


def train_metrics_debiased(target_batchsize, seed=0, plot=False,lambda_ = 1):
    lr = 5e-4
    drug = data_loader.Drug(True)
    model = models.NeuralNetworkClassification(drug.get_k())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    fairloss = fairness_metrics.energy_distance

    data_loader.set_seed(seed)
    drug.split_test()
    X, Y, A = drug.get_data()
    X_test, Y_test, A_test = drug.get_test_data()
    
    N_epochs = 500

    losses = []
    test_losses = []
    for epoch in range(N_epochs):
        X,Y,A = shuffle(X,Y,A)
        sumloss = 0
        batchstart = 0
        while batchstart<len(A):
            optimizer.zero_grad()
            batchsize = find_batchsize(target_batchsize, A[batchstart:])
            if batchsize>0:
                X_batch, Y_batch, A_batch = X[batchstart:batchstart+batchsize], Y[batchstart:batchstart+batchsize], A[batchstart:batchstart+batchsize]
                batchstart = batchstart+batchsize
                pred = model(X_batch)
                #L = criterion(pred, Y_batch)
                y_after_sig = torch.sigmoid(pred)
                y_after_sig = y_after_sig[:, None]
                y_hat_1 = pred[A_batch.flatten()==1]
                y_hat_0 = pred[A_batch.flatten()==0]
                L_fair = fairloss(y_hat_1, y_hat_0)
                # overall loss
                y_1 = Y_batch[A_batch.flatten()==1]
                y_0 = Y_batch[A_batch.flatten()==0]
                delta_1, delta_0 = 1, 1
                N = len(A_batch)
                N_1 = A_batch.sum()
                N_0 = N-N_1
                if N >= target_batchsize:
                    if N_1 == 2:
                        delta_1 = N/(2*(N-1))
                        delta_0 = N/((N-1))
                    else:
                        delta_1 = N/((N-1))
                        delta_0 = N/(2*(N-1))
                weight_1 = (delta_1) * N_1/N
                weight_0 = (delta_0) * N_0/N
                accloss1 = criterion(y_hat_1, y_1)
                accloss0 = criterion(y_hat_0, y_0)
                L = (weight_0 * accloss0 + weight_1 * accloss1)
                loss = L + lambda_ * L_fair
                loss.backward()
                sumloss += loss.detach().item()
                optimizer.step()
            else:
                batchstart = len(A)+1
        losses.append(sumloss)
        with torch.no_grad():
            pred = model(X_test)
            y_hat_1 = pred[A_test.flatten()==1]
            y_hat_0 = pred[A_test.flatten()==0]
            testloss = criterion(model(X_test), Y_test) + fairloss(y_hat_1, y_hat_0)
            test_losses.append(testloss)

    if plot:
        plt.plot(losses)

    y_hat = torch.round(torch.sigmoid(model(X_test)))
    y_hat_1 = y_hat[A_test.flatten()==1].flatten()
    y_hat_0 = y_hat[A_test.flatten()==0].flatten()
    y_1 = Y_test[A_test.flatten()==1].flatten()
    y_0 = Y_test[A_test.flatten()==0].flatten()

    test_results = {}

    for key in metrics.keys():
        test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()
    return test_results, losses, test_losses


def train_metrics_noreg(target_batchsize, seed=0, plot=False):
    lr = 5e-4
    drug = data_loader.Drug(True)
    model = models.NeuralNetworkClassification(drug.get_k())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    fairloss = fairness_metrics.energy_distance_biased

    data_loader.set_seed(seed)
    drug.split_test()
    X, Y, A = drug.get_data()
    X_test, Y_test, A_test = drug.get_test_data()
    
    N_epochs = 500

    losses = []
    test_losses = []
    for epoch in range(N_epochs):
        X,Y,A = shuffle(X,Y,A)
        sumloss = 0
        batchstart = 0
        while batchstart<len(A):
            optimizer.zero_grad()
            batchsize = min(target_batchsize, len(A[batchstart:]))
            if batchsize>0:
                X_batch, Y_batch, A_batch = X[batchstart:batchstart+batchsize], Y[batchstart:batchstart+batchsize], A[batchstart:batchstart+batchsize]
                batchstart = batchstart+batchsize
                pred = model(X_batch)
                L = criterion(pred, Y_batch)
                # overall loss
                loss = L
                loss.backward()
                sumloss += loss.detach().item()
                optimizer.step()
            else:
                batchstart = len(A)+1
        losses.append(sumloss)
        with torch.no_grad():
            testloss = criterion(model(X_test), Y_test)
            test_losses.append(testloss)

    if plot:
        plt.plot(losses)

    y_hat = torch.round(torch.sigmoid(model(X_test)))
    y_hat_1 = y_hat[A_test.flatten()==1].flatten()
    y_hat_0 = y_hat[A_test.flatten()==0].flatten()
    y_1 = Y_test[A_test.flatten()==1].flatten()
    y_0 = Y_test[A_test.flatten()==0].flatten()

    test_results = {}

    for key in metrics.keys():
        test_results[key] = metrics[key](y_hat_1, y_hat_0, y_1, y_0).data.item()
    return test_results, losses, test_losses