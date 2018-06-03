import pandas as pd
import numpy as np
import random
from collections import Counter

'''
INCLUDED FUNCTIONS
add_kfold_indexes(K,Data)
train_test_merge(train,test,label)
cat_to_int(data, cat_to_int_cols)
'''

def add_kfold_indexes(Data, K):
    random.seed(a=42)
    indices = list(range(len(Data)))
    folds = [-1]*len(Data)
    k = -1
    for rep in range(len(folds)):
        k = (k + 1) % K
        index_of_index = random.randint(0, len(indices)-1)
        folds[indices[index_of_index]] = k
        del indices[index_of_index]
    #print(Counter(folds))
    pd.options.mode.chained_assignment = None
    Data["Fold"] = folds
    pd.options.mode.chained_assignment = 'warn'
    return Data

def train_test_merge(train, test, label):
    test[label] = "None"
    train["IsTrain"] = True
    test["IsTrain"] = False
    train_test = pd.concat([train,test])
    return(train_test)

def train_test_split(train_test):
    train = train_test[train_test["IsTrain"] == True]
    test = train_test[train_test["IsTrain"] == False]
    train.drop(["IsTrain"], axis=1)
    test.drop(["IsTrain"], axis=1)
    return train, test

def cat_to_int(data, cat_to_int_cols):
    for col in cat_to_int_cols:
        cats = list(data[col].unique())
        data[col] = data[col].apply(cats.index)
    return data

def kfold_split(data, k, label, features):
    x_train_train = data[data["Fold"] != k][features].values
    y_train_train = data[data["Fold"] != k][label].values
    x_train_test = data[data["Fold"] == k][features].values
    y_train_test = data[data["Fold"] == k][label].values
    return x_train_train, y_train_train, x_train_test, y_train_test
