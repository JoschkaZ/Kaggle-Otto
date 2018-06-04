import pandas as pd
import numpy as np
import random
from collections import Counter
from paths import PATH

'''
INCLUDED FUNCTIONS
add_kfold_indexes(K,Data)
train_test_merge(train,test,label)
cat_to_int(data, cat_to_int_cols)
'''

def get_data(label, no_feature_columns, cat_to_int_columns,
            cat_to_onehot_columns, K):

    #READ DATA
    train=pd.read_csv(PATH + r'train.csv')
    test=pd.read_csv(PATH + r'test.csv')

    #GET FEATURES
    features = list(train.columns.values)
    features = [x for x in features if x not in no_feature_columns]

    #DATA TRANSFORMATIONS
    train, test = cat_to_int(train, test, label, cat_to_int_columns)
    train, test, features, labels = cat_to_onehot(train, test, features, label, cat_to_onehot_columns)



    train = add_kfold_indexes(train, K)


    return train, test, features, labels

def cat_to_int(train, test, label, cat_to_int_cols):
    for col in cat_to_int_cols:
        if col == label:
            cats = list(train[col].unique())
            train[col] = train[col].apply(cats.index)
        else:
            train_test = train_test_merge(train,test,label)
            cats = list(train_test[col].unique())
            train_test[col] = train_test[col].apply(cats.index)
            train, test = train_test_split(train_test)
    return train, test

def cat_to_onehot(train, test, features, label, cat_to_onehot_columns):
    if len(cat_to_onehot_columns) > 0:
        print(cat_to_onehot_columns)
        for col in cat_to_onehot_columns:
            if col == label:
                f1 = list(train.columns.values)
                train = pd.get_dummies(train, columns = [col])
                labels  = list(set(list(train.columns.values)) - set(f1))
            else:
                f1 = list(train.columns.values)
                train_test = train_test_merge(train,test,label)
                train_test = pd.get_dummies(train_test, columns = [col])
                train, test = train_test_split(train_test)
                labels = [label]
                features.extend(list(set(list(train.columns.values)) - set(f1)))
                features.remove(col)
                print("F: ", features)
        return train, test, features, labels
    else: return train, test, features, [label]


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





def kfold_split(data, k, label, features):
    x_train_train = data[data["Fold"] != k][features].values
    y_train_train = data[data["Fold"] != k][label].values
    x_train_test = data[data["Fold"] == k][features].values
    y_train_test = data[data["Fold"] == k][label].values
    return x_train_train, y_train_train, x_train_test, y_train_test
