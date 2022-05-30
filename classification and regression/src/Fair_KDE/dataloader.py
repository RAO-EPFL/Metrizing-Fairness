import os
import copy
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_loader
from tempeh.configurations import datasets
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder, StandardScaler


def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(device), torch.FloatTensor(XZ).to(device)

def adult(data_root, display=False):
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

def compas_data_loader():
    """ Downloads COMPAS data from the propublica GitHub repository.
    :return: pandas.DataFrame with columns 'sex', 'age', 'juv_fel_count', 'juv_misd_count',
       'juv_other_count', 'priors_count', 'two_year_recid', 'age_cat_25 - 45',
       'age_cat_Greater than 45', 'age_cat_Less than 25', 'race_African-American',
       'race_Caucasian', 'c_charge_degree_F', 'c_charge_degree_M'
    """
    data = pd.read_csv("./data/compas/compas-scores-two-years.csv")  # noqa: E501
    # filter similar to
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    data = data[(data['days_b_screening_arrest'] <= 30) &
                (data['days_b_screening_arrest'] >= -30) &
                (data['is_recid'] != -1) &
                (data['c_charge_degree'] != "O") &
                (data['score_text'] != "N/A")]
    # filter out all records except the ones with the most common two races
    data = data[(data['race'] == 'African-American') | (data['race'] == 'Caucasian')]
    # Select relevant columns for machine learning.
    # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
    data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                 "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]]
    # map string representation of feature "sex" to 0 for Female and 1 for Male
    data = data.assign(sex=(data["sex"] == "Male") * 1)
    data = pd.get_dummies(data)
    return data


class CustomDataset():
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z


class FairnessDataset():
    def __init__(self, dataset, device=torch.device('cuda')):
        self.dataset = dataset
        self.device = device
        np.random.seed(12345678)
        
        if self.dataset == 'AdultCensus':
            self.get_adult_data()
        elif self.dataset == 'COMPAS':
            self.get_compas_data()
        elif self.dataset == 'CreditDefault':
            self.get_credit_default_data()
        elif self.dataset == 'Lawschool':
            self.get_lawschool_data()
        elif self.dataset == 'Moon':
            self.get_moon_data()
        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))
        self.prepare_ndarray()
    
    def get_adult_data(self):
        X_train, Y_train, X_test, Y_test = adult('./data/adult/')
        
        self.Z_train_ = X_train['Sex']
        self.Z_test_ = X_test['Sex']
        self.X_train_ = X_train.drop(labels=['Sex'], axis=1)
        self.X_train_ = pd.get_dummies(self.X_train_)
        self.X_test_ = X_test.drop(labels=['Sex'], axis=1)
        self.X_test_ = pd.get_dummies(self.X_test_)

        le = LabelEncoder()
        self.Y_train_ = le.fit_transform(Y_train)
        self.Y_train_ = pd.Series(self.Y_train_, name='>50k')
        self.Y_test_ = le.fit_transform(Y_test)
        self.Y_test_ = pd.Series(self.Y_test_, name='>50k')
#     def get_compas_data(self):
#         dataset = datasets['compas']()
#         # dataset = compas_data_loader()
#         X_train, X_test = dataset.get_X(format=pd.DataFrame)
#         Y_train, Y_test = dataset.get_y(format=pd.Series)
#         Z_train, Z_test = dataset.get_sensitive_features('race', format=pd.Series)

#         self.X_train_ = X_train
#         self.Y_train_ = Y_train
#         self.Z_train_ = (Z_train != 'African-American').astype(float)

#         self.X_test_ = X_test
#         self.Y_test_ = Y_test
#         self.Z_test_ = (Z_test != 'African-American').astype(float)
    def get_compas_data(self):
        dataset = datasets['compas']()
        ds = data_loader.Compas()
        ds.split_test()
        X, Y, A = ds.get_log_data()
        X_test, Y_test, A_test = ds.get_test_data()


        self.X_train_ = X
        self.Y_train_ = Y
        self.Z_train_ = A

        self.X_test_ = X_test
        self.Y_test_ = Y_test
        self.Z_test_ = A_test

    def get_credit_default_data(self):
        rawdata = pd.read_excel('./data/credit_card/default_clients.xls', header=1)
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
        Z = 2 - preproc_data['SEX']
        
        self.X_train_ = X.loc[list(range(24000)), :]
        self.Y_train_ = Y.loc[list(range(24000))]
        self.Z_train_ = Z.loc[list(range(24000))]
        
        self.X_test_ = X.loc[list(range(24000,30000)), :]
        self.Y_test_ = Y.loc[list(range(24000,30000))]
        self.Z_test_ = Z.loc[list(range(24000,30000))]
    
    def get_lawschool_data(self):
        rawdata = pd.read_sas('./data/lawschool/lawschs1_1.sas7bdat')
        rawdata = rawdata.drop(['college', 'Year', 'URM', 'enroll'], axis=1)
        rawdata = rawdata.dropna(axis=0)
        rawdata = rawdata.sample(frac=1.0, random_state=12345678).reset_index(drop=True)
        
        X = rawdata[['LSAT', 'GPA', 'Gender', 'resident']]
        Y = rawdata['admit']
        Z = rawdata['White']
        
        self.X_train_ = X.loc[list(range(77267)), :]
        self.Y_train_ = Y.loc[list(range(77267))]
        self.Z_train_ = Z.loc[list(range(77267))]
        
        self.X_test_ = X.loc[list(range(77267,96584)), :]
        self.Y_test_ = Y.loc[list(range(77267,96584))]
        self.Z_test_ = Z.loc[list(range(77267,96584))]
    
    def get_moon_data(self):
        n_train = 10000
        n_test = 5000
        X, Y = make_moons(n_samples=n_train+n_test, noise=0.2, random_state=0)
        Z = np.zeros_like(Y)

        np.random.seed(0)
        for i in range(n_train + n_test):
            if Y[i] == 0:
                if -0.734 < X[i][0] < 0.734:
                    Z[i] = np.random.binomial(1, 0.90)
                else:
                    Z[i] = np.random.binomial(1, 0.35)
            elif Y[i] == 1:
                if 0.262 < X[i][0] < 1.734:
                    Z[i] = np.random.binomial(1, 0.55)
                else:
                    Z[i] = np.random.binomial(1, 0.10)

        X = pd.DataFrame(X, columns=['x_1', 'x_2'])
        Y = pd.Series(Y, name='label')
        Z = pd.Series(Z, name='sensitive attribute')
        
        self.X_train_ = X.loc[list(range(10000)), :]
        self.Y_train_ = Y.loc[list(range(10000))]
        self.Z_train_ = Z.loc[list(range(10000))]
        
        self.X_test_ = X.loc[list(range(10000,15000)), :]
        self.Y_test_ = Y.loc[list(range(10000,15000))]
        self.Z_test_ = Z.loc[list(range(10000,15000))]
    
    def prepare_ndarray(self):
        self.normalized = False
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None
        
    def normalize(self):
        self.normalized = True
        scaler_XZ = StandardScaler()
        self.XZ_train = scaler_XZ.fit_transform(self.XZ_train)
        self.XZ_test = scaler_XZ.transform(self.XZ_test)
        
        scaler_X = StandardScaler()
        self.X_train = scaler_X.fit_transform(self.X_train)
        self.X_test = scaler_X.transform(self.X_test)
        return None
    
    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train),\
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test)
    
    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)