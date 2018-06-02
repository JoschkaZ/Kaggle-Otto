import pandas as pd
import numpy as np
import random
from collections import Counter

data=pd.read_csv(r'C:\Data\Otto\train.csv')
#data=pd.read_csv(r'C:\Data\Otto\test.csv')




def add_kfold_indexes(K, Data):
    '''
    Adds randomly distributed Kfold indexes as a new column to the data frame. All folds will have the same size if possible.
    Inputs:
    K - Number of folds to split into
    Data - Pandas dataframe
    Outputs:
    Data - Pandas dataframe with new group indices added as column "Fold"
    '''
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



Data = add_kfold_indexes(5,data)
print(Data)
