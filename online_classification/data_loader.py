# data_loader.py
# utilities for loading data
import torch
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from load_data import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing

"""
% Metrizing Fairness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script provides data loading functinality for MFL.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def to_tensor(data, device):
    D = data
    if type(data) == pd.core.frame.DataFrame:
        D = data.to_numpy()

    if type(D) == np.ndarray:
        return torch.tensor(D, device=device).float()
    elif type(D) == torch.Tensor:
        return D.to(device).float()
    else:
        raise NotImplementedError('Currently only Torch Tensors, Numpy NDArrays and Pandas Dataframes are supported')


class DataLoader:
    def __init__(self, X, Y, A, X_test=None, Y_test=None, A_test=None, use_tensor=True, device='cpu', info='No Info Available', min_max_scaler=None):
        self.device = device
        self.use_tensor = use_tensor
        self.X = to_tensor(X, device) if use_tensor else X
        self.A = to_tensor(A, device) if use_tensor else A
        self.Y = to_tensor(Y, device) if use_tensor else Y
        if X_test is not None:
            self.X_test = to_tensor(X_test, device) if use_tensor else X_test
            self.A_test = to_tensor(A_test, device) if use_tensor else A_test
            self.Y_test = to_tensor(Y_test, device) if use_tensor else Y_test
        self.info = info
        self.min_max_scaler = None


    def get_data(self):
        # get the dataset
        return (self.X, self.Y, self.A)
    
    def get_adult_data(self):
        return (self.X, self.Y, self.A, self.X_test, self.Y_test, self.A_test)

    def get_data_for_A(self, a):
        # get dataset but only for samples with attribute a
        X_a = self.X[(self.A==a).squeeze()]
        Y_a = self.Y[(self.A==a).squeeze()]
        return (X_a, Y_a)
    
    def stratified_batch_generator_worep(self, batch_size=32, n_epochs=100):
        # get propoertions of protected attribute
        p_A1 = self.A.mean()
        p_A0 = 1 - p_A1
        total_samples = self.A.shape[0]
        # build index set of protected and unprotected attribute

        # number of samples to sample from each distribution
        n_batch_1 = int(p_A1*batch_size)
        n_batch_0 = int(p_A0*batch_size)

        for epoch in tqdm(range(n_epochs)):
            ind_A1 = (self.A==1).nonzero()[:,0]
            ind_A0 = (self.A==0).nonzero()[:,0]
            for _ in range(0, total_samples - batch_size + 1, batch_size):
                # sample indexes for protected and unprotected class
                sampled_indices_A1 = (torch.ones(ind_A1.shape[0]) / (ind_A1.shape[0])).multinomial(
                                                                                    num_samples=n_batch_1, 
                                                                                    replacement=False)
                batch_idx1 = ind_A1[sampled_indices_A1]
                mask = torch.ones(ind_A1.numel(), dtype=torch.bool)
                mask[sampled_indices_A1] = False
                ind_A1 = ind_A1[mask]

                sampled_indices_A0 = (torch.ones(ind_A0.shape[0]) / (ind_A0.shape[0])).multinomial(
                                                                                    num_samples=n_batch_0, 
                                                                                    replacement=False)
                batch_idx0 = ind_A0[sampled_indices_A0]
                mask = torch.ones(ind_A0.numel(), dtype=torch.bool)
                mask[sampled_indices_A0] = False
                ind_A0 = ind_A0[mask]
                
                yield (torch.vstack((self.X[batch_idx0], self.X[batch_idx1])),
                   torch.vstack((self.Y[batch_idx0], self.Y[batch_idx1])),
                   torch.vstack((self.A[batch_idx0], self.A[batch_idx1])))
    
    def get_info(self):
        return self.info

    def split_test(self, **kwargs):
        # perform train test split, kwargs for sklearn train-test-split
        X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(self.X, self.Y, self.A, **kwargs)
        self.X = X_train
        self.X_test = X_test
        self.Y = Y_train
        self.Y_test = Y_test
        self.A = A_train
        self.A_test = A_test

    def get_test_data(self):
        # get the test dataset
        if self.X_test is None:
            raise ValueError('Train-Test split has not yet been performed')
        if self.min_max_scaler is not None:
            x_vals = self.X_test.values #returns a numpy array
            x_scaled = self.min_max_scaler.fit_transform(x_vals)
            self.X_test = pd.DataFrame(x_scaled)
            
        return (self.X_test, self.Y_test, self.A_test)

    def get_log_data(self):
        # get the dataset
        return (self.X, self.Y, self.A)

    def get_k(self):
        return self.X.shape[1]

class LawSchool(DataLoader):
    """This is a classification dataset"""
    def __init__(self, a_inside_x, **kwargs):
        rawdata = pd.read_sas('./data/classification/lawschool/lawschs1_1.sas7bdat')
        rawdata = rawdata.drop(['college', 'Year', 'URM', 'enroll'], axis=1)
        rawdata = rawdata.dropna(axis=0)
        rawdata = rawdata.sample(frac=1.0, random_state=12345678).reset_index(drop=True)

        X = rawdata[['LSAT', 'GPA', 'Gender', 'resident']]
        Y = rawdata['admit']
        A = rawdata['White']
        if a_inside_x:
            X = pd.concat((X, A), axis=1)
        info = '''Law School Admissions Data collected by Project SEAPHE, predict admission, 
        don\'t discriminate White vs. Non-White\nhttp://www.seaphe.org/databases.php'''

        super().__init__(X, np.array(Y)[:, None], np.array(A)[:, None], info=info, **kwargs)
      
class Drug(DataLoader):
    """This is a classification dataset"""
    def __init__(self, a_inside_x):
        X, Y, A = load_drug_data('data/classification/drug/drug_consumption.data.txt')
        info = '''
        https://www.kaggle.com/danofer/compass
        '''
        if a_inside_x:
            X = np.concatenate((X, A[:, None]), axis=1)
        super().__init__(X, Y[:, None], A[:, None], info=info)
    
class Credit(DataLoader):
    """This is a classification dataset"""
    def __init__(self, a_inside_x, **kwargs): 
        rawdata = pd.read_excel('./data/classification/credit_card/default_clients.xls', header=1)
        rawdata = rawdata.sample(frac=1.0, random_state=12345678).reset_index(drop=True)

        columns = list(rawdata.columns)
        categ_cols = []
        for column in columns:
            if 2 < len(set(rawdata[column])) < 10:
                categ_cols.append((column, len(set(rawdata[column]))))

        preproc_data = copy.deepcopy(rawdata)
        for categ_col, n_items in categ_cols:
            for i in range(n_items):
                preproc_data[categ_col + str(i)] = (preproc_data[categ_col] == i).astype(float)
        preproc_data = preproc_data.drop(['EDUCATION', 'MARRIAGE'], axis=1)

        X = preproc_data.drop(['ID', 'SEX', 'default payment next month'], axis=1)
        Y = preproc_data['default payment next month']
        A = 2 - preproc_data['SEX']
        
        if a_inside_x:
            X = pd.concat((X, A), axis=1)
        info = '''Credit data'''
        
        self.min_max_scaler = preprocessing.MinMaxScaler()

        x_vals = X.values #returns a numpy array
        x_scaled = self.min_max_scaler.fit_transform(x_vals)
        X = pd.DataFrame(x_scaled)


        super().__init__(X, np.array(Y)[:, None], np.array(A)[:, None], info=info, **kwargs)
     
class Adult(DataLoader):
     def __init__(self, a_inside_x, **kwargs):
        X_train_, Y_train_, X_test_, Y_test_ = adult_data_read('./data/classification/adult/')

        A = X_train_['Sex']
        A_test = X_test_['Sex']

        le = LabelEncoder()
        Y = le.fit_transform(Y_train_)
        Y = pd.Series(Y, name='>50k')
        Y_test = le.fit_transform(Y_test_)
        Y_test = pd.Series(Y_test, name='>50k')
        
        if not a_inside_x:
            X = X_train_.drop(labels=['Sex'], axis=1)
            X = pd.get_dummies(X)
            X_test = X_test_.drop(labels=['Sex'], axis=1)
            X_test = pd.get_dummies(X_test)
        else:
            X = pd.get_dummies(X_train_)
            X_test = pd.get_dummies(X_test_)

        info = """Adult dataset for classification. Train Test split is already provided"""
        super().__init__(X, np.array(Y)[:, None], np.array(A)[:, None], X_test, np.array(Y_test)[:, None], np.array(A_test)[:, None], info=info, **kwargs)
      
class CommunitiesCrime(DataLoader):
    # http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
    def __init__(self, **kwargs):
        yvar = 'ViolentCrimesPerPop'
        avar = 'racepctblack'
        # load the data
        with open('data/communities.names') as file:
            info = file.read()
        colnames = [line.split(' ')[1] for line in info.split('\n') if line and line.startswith('@attribute')]
            
        df = pd.read_csv('data/communities.data', 
                        header=None, 
                        names=colnames,
                        na_values='?')

        # process the data
        Y = df[[yvar]]
        A = (df[[avar]] > df[[avar]].median()).astype(int)
        nasum = df.isna().sum()
        names = [name for name in nasum[nasum==0].index if name not in [yvar, avar, 'state', 'communityname', 'fold']]
        X = df[names]
        # init super
        super().__init__(X, Y, A, info=info, **kwargs)
        
class CommunitiesCrimeClassification(DataLoader):
    # http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
    def __init__(self, a_inside_x, **kwargs):
        yvar = 'ViolentCrimesPerPop'
        avar = 'racepctblack'
        # load the data
        with open('data/communities.names') as file:
            info = file.read()
        colnames = [line.split(' ')[1] for line in info.split('\n') if line and line.startswith('@attribute')]
            
        df = pd.read_csv('data/communities.data', 
                        header=None, 
                        names=colnames,
                        na_values='?')

        # process the data
        Y = df[[yvar]]
        bin_thr = Y.mean()
        Y = (Y>= bin_thr).astype(int)
        A = (df[[avar]] > df[[avar]].median()).astype(int)
        nasum = df.isna().sum()
        names = [name for name in nasum[nasum==0].index if name not in [yvar, avar, 'state', 'communityname', 'fold']]
        X = df[names]
        if a_inside_x:
            X = np.concatenate((np.array(X), np.array(A)), axis=1)
        # init super
        super().__init__(X, Y, A, info=info, **kwargs)

class BarPass(DataLoader):
    # http://www.seaphe.org/databases.php
    def __init__(self, **kwargs):
        df = pd.read_sas('data/lawschs1_1.sas7bdat')
        drop_cols = ['enroll', 'college', 'Year', 'Race']
        df = df[[col for col in df.columns if col not in drop_cols]]
        df = df.dropna()
        Y = df[['GPA']]
        A = df[['White']]
        X = df.drop('GPA', axis=1)
        info = '''Law School Admissions Data collected by Project SEAPHE, predict GPA, 
        don\'t discriminate White vs. Non-White\nhttp://www.seaphe.org/databases.php'''
        self.first_call = True
        super().__init__(X, Y, A, info=info, **kwargs)

    def get_data(self):
        if self.first_call:
            self.Xs, self.Ys, self.As = next(self.stratified_batch_generator_worep(500, 1))
            self.first_call = False
        return (self.Xs, self.Ys, self.As)
    
    def get_log_data(self):
        return self.get_data()

class StudentPerformance(DataLoader):
    # https://archive.ics.uci.edu/ml/datasets/student+performance
    def __init__(self, subject = 'Math', **kwargs):
        # load data
        df = pd.read_csv('data/student/student-{}.csv'.format(subject.lower()[:3]), sep=';')\
        # convert the categorical values
        categoricals = df.dtypes[df.dtypes==object].index
        for attribute in categoricals:
            options = df[attribute].unique()
            options.sort()
            options = options[:-1]
            for option in options:
                df['{}_{}'.format(attribute, option)] = (df[attribute]==option).astype(int)
            df = df.drop(attribute, axis=1)
        # extract X A Y
        A = df[['sex_F']]
        Y = df[['G3']]
        X = df.drop(['sex_F', 'G3'], axis=1)
        info = '''
        Student Performance dataset. Predict Final Grade based on Attributes, don't discriminate against female students.
        https://archive.ics.uci.edu/ml/datasets/student+performance
        '''
        super().__init__(X, Y, A, info=info, **kwargs)
        
class Compas(DataLoader):
    def __init__(self, a_inside_x):
        X, Y, A = load_compas_data('data/classification/compas/compas-scores-two-years.csv')
        info = '''
        https://www.kaggle.com/danofer/compass
        '''
        if a_inside_x:
            X = np.concatenate((X, A[:, None]), axis=1)
        super().__init__(X, Y[:, None], A[:, None], info=info)
      

class Synthetic1(DataLoader):
    # synthetic data: bias offset
    def __init__(self, N, k, delta_intercept = 0.5, **kwargs):
        X_0 = torch.normal(mean=0.0, std=torch.ones(int(N/2),k))
        X_1 = X_0
        theta = torch.normal(mean=2, std=torch.ones(k,1))
        Y_0 = delta_intercept+ X_0@theta
        Y_1 = X_1@theta
        A_0 = torch.zeros(int(N/2),1)
        A_1 = torch.ones(N-int(N/2),1)
        info = 'Synthetic Data'
        X = torch.vstack((X_0, X_1))
        Y = torch.vstack((Y_0, Y_1))
        A = torch.vstack((A_0, A_1))
        super().__init__(np.hstack((X,A)),
                         Y,
                         A, 
                         info=info, **kwargs)

class Synthetic2(DataLoader):
    # synthetic data: bias slope
    def __init__(self, N, k, delta_slope = 0.5, **kwargs):
        X_0 = torch.normal(mean=0.0, std=torch.ones(int(N/2),k))
        X_1 = X_0
        theta = torch.normal(mean=2, std=torch.ones(k,1))
        Y_0 = X_0@(theta+delta_slope)
        Y_1 = X_1@theta
        A_0 = torch.zeros(int(N/2),1)
        A_1 = torch.ones(N-int(N/2),1)
        info = 'Synthetic Data'
        X = torch.vstack((X_0, X_1))
        Y = torch.vstack((Y_0, Y_1))
        A = torch.vstack((A_0, A_1))
        super().__init__(np.hstack((X,A)),
                         Y,
                         A, 
                         info=info, **kwargs)

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
  
def adult_data_read(data_root, display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_train_data = pd.read_csv(
        data_root+'adult.data',
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    raw_test_data = pd.read_csv(
        data_root+'adult.test',
        skiprows=1,
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    train_data = raw_train_data.drop(["Education"], axis=1)  # redundant with Education-Num
    test_data = raw_test_data.drop(["Education"], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
    train_data["Target"] = train_data["Target"] == " >50K"
    test_data["Target"] = test_data["Target"] == " >50K."
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                train_data[k] = np.array([rcode[v.strip()] for v in train_data[k]])
                test_data[k] = np.array([rcode[v.strip()] for v in test_data[k]])
            else:
                train_data[k] = train_data[k].cat.codes
                test_data[k] = test_data[k].cat.codes

    return train_data.drop(["Target", "fnlwgt"], axis=1), train_data["Target"].values, test_data.drop(["Target", "fnlwgt"], axis=1), test_data["Target"].values
