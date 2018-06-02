import pandas as pd
import numpy as np
import random
from collections import Counter

data=pd.read_csv(r'C:\Data\Otto\train.csv')
#data=pd.read_csv(r'C:\Data\Otto\test.csv')




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
    train["TrainTest"] = "Train"
    test["TrainTest"] = "Test"

    train_test = pd.concat([train,test])
    return(train_test)

def cat_to_int(data, cat_to_int_cols):
    for col in cat_to_int_cols:
        cats = list(data[col].unique())
        print(cats)
        print(cats.index('Class_4'))
        print(data[col].apply(cats.index))
        data[col] = data[col].apply(cats.index)
    return data
