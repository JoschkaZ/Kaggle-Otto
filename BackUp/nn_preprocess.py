import numpy as np
import pandas as pd
from paths import PATH

CLASS_COUNT = 9
FEATURE_COUNT = 93
TARGET_COL = "target"

def cat_to_int(col):
    cats = list(col.unique())
    return col.apply(cats.index)

#formats into one-hot-vector
def get_targets(data):
    temp = data[TARGET_COL]
    temp = cat_to_int(temp)
    targets = np.zeros(shape=(len(temp),9))
    for i, value in enumerate(temp):
        targets[i,value] = 1
    return targets

def max_index(a):
    temp = 0
    temp_idx = 0
    for i,x in enumerate(a):
        if x > temp:
            temp = x
            temp_idx = i
    return temp_idx

def one_hot_inverse(data):
    result = []
    for i, line in enumerate(data):
        result.append(int(max_index(line)))
    return np.array(result, dtype=int)

def get_train():
    train = pd.read_csv(PATH+"train.csv")
    return np.array(train)[:,1:-1], get_targets(train)

def get_test():
    test = pd.read_csv(PATH + "test.csv")
    return np.array(test)[:,1:]
