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
### INCLUDED FUNCTIONS ###



def add_kfold_indexes(K, Data):
    """
    :param K:  Number of folds to split into
    :param Data: Pandas Dataframe

    :return: Pandas dataframe with new Group Indices added as column "Fold"
    """
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
    Data["Fold"] = folds
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
    return train, test

def cat_to_int(data, cat_to_int_cols):
    for col in cat_to_int_cols:
        cats = list(data[col].unique())
        data[col] = data[col].apply(cats.index)
    return data

def kfold_split(data, k, label):
    x_train_train = data[data["Fold"] != k]
    y_train_train = data[data["Fold"] != k][LABEL]
    x_train_test = data[data["Fold"] == k]
    y_train_test = data[data["Fold"] == k][LABEL]

    x_train_train.drop([LABEL, "FOLD"], axis = 1)
    x_train_test.drop([LABEL, "FOLD"], axis = 1)

    return x_train_train, y_train_train, x_train_test, x_train_train
