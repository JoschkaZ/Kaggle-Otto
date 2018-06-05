import pandas as pd
import numpy as np
import random
from collections import Counter
from paths import PATH

'''
INCLUDED FUNCTIONS

get_data(label, no_feature_columns, cat_to_int_columns, cat_to_onehot_columns, K)

train_test_merge(train,test,label)
train_test_split(train_test)

cat_to_int(train, test, label, cat_to_int_cols)
cat_to_onehot(train, test, features, label, cat_to_onehot_columns)

add_kfold_indexes(Data, K)
kfold_split(data, k, labels, features)

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
    #KFold Indexes
    train = add_kfold_indexes(train, K)
    return train, test, features, labels

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
    pd.options.mode.chained_assignment = None
    Data["Fold"] = folds
    pd.options.mode.chained_assignment = 'warn'
    return Data

def kfold_split(data, k, labels, features):
    x_train_train = data[data["Fold"] != k][features].values
    y_train_train = data[data["Fold"] != k][labels].values
    x_train_test = data[data["Fold"] == k][features].values
    y_train_test = data[data["Fold"] == k][labels].values
    return x_train_train, y_train_train, x_train_test, y_train_test

class Data:
    def __init__(self, labels : list, cat_to_int_columns, fold_count = 5, no_feature_columns = list()):
        self.train=pd.read_csv(PATH + r'train.csv')
        self.test=pd.read_csv(PATH + r'test.csv')

        #GET FEATURES
        self.features = list(self.train.columns.values)
        self.features = [x for x in self.features if x not in no_feature_columns]
        self.fold_count = fold_count
        self.labels = labels
        self.cat_to_int_columns = cat_to_int_columns

        for col in cat_to_int_columns:
            self.cat_to_int(col)
        self.add_fold_indexes(fold_count)

        self.x_train  = self.train[self.features]
        self.y_train = self.train[self.labels]
        self.x_test = self.test[self.features]

    def get_label_shape(self):
        return self.y_train.shape[1]

    def get_label_columns(self):
        return self.y_train.columns

    def get_feature_shape(self):
        return self.x_train.shape[1]

    def get_feature_columns(self):
        return self.x_train.columns

    def cat_to_int(self, col):
        index = list(self.train[col].unique()).index
        to_int = lambda data, c: data[c].apply(index)
        if col in self.labels:
            self.train[col] = to_int(self.train, col)
        else:
            self.train[col], self.test[col] = to_int(self.train, col), to_int(self.test, col)

    def add_fold_indexes(self, count, shuffle = True):
        repeats = self.train.shape[0] // count +1
        offset = count-(self.train.shape[0] % count)
        folding = np.tile(np.arange(count),repeats)
        if offset != count:
            folding = folding[:-offset]
        if shuffle:
            np.random.shuffle(folding)
        self.folding = folding

    def fold_split(self, k):
        x_train_train = self.x_train[self.folding != k].values
        y_train_train = self.y_train[self.folding != k].values
        x_train_test = self.x_train[self.folding == k].values
        y_train_test = self.y_train[self.folding == k].values
        return x_train_train, y_train_train, x_train_test, y_train_test

    def max_index(self, a):
        temp = 0
        temp_idx = 0
        for i,x in enumerate(a):
            if x > temp:
                temp = x
                temp_idx = i
        return temp_idx

    def inverse_onehot(self, data):
        result = []
        for i, line in enumerate(data):
            result.append(int(self.max_index(line)))
        return np.array(result, dtype=int)

    def to_onehot(self,data, dim):

        #print(str(data.shape)+ " "+ str(dim))
        targets = np.zeros(shape=(len(data),dim))
        for i, value in enumerate(data):
            targets[i,int(value)] = 1
        return pd.DataFrame(targets)

    def get_train(self):
        return self.x_train, self.y_train

    #deprecated
    def col_to_onehot(self, col):
        if not col in self.x_train and not col in self.y_train:
            print("Data.col_to_onehot: no such column")
            return
        if col in self.x_train:
            dim = max(max(self.x_train[col]), max(self.x_test[col])) +1
            onehot = self.to_onehot(self.x_train[col],dim)
            self.x_train = pd.concat([self.x_train,onehot],axis =1)
            onehot = self.to_onehot(self.x_test[col],dim)
            self.x_test = pd.concat([self.x_test,onehot],axis=1)
            self.x_train = self.x_train.drop(col,axis=1)
            self.x_test = self.x_test.drop(col,axis=1)
        if col in self.y_train:
            dim = int(max(self.y_train[col])) +1
            onehot = self.to_onehot(self.y_train[col],dim)
            self.y_train = pd.concat([self.y_train,onehot], axis=1)
            self.y_train = self.y_train.drop(col,axis=1)
