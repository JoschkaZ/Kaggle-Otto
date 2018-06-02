import numpy as np
import pandas as pd
from PreProcess import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

PATH = r'C:\Data\Otto'
LABEL = 'target'
K = 5
CAT_TO_INT_COLS = ['target']



train=pd.read_csv(PATH + r'\train.csv')
test=pd.read_csv(PATH + r'\test.csv')


train = add_kfold_indexes(K,train)

train = cat_to_int(train, CAT_TO_INT_COLS)


print(train)
#print(pd.factorize(train[LABEL])[0])
'''

rd= RandomForestClassifier(n_estimators=100)

for k in range(K):

    x_train = train[train["Fold"] != k]
    y_train = train[train["Fold"] == k]
    display(x_train.head(5))
    display(y_train.head(5))

    rd.fit(x_train,y_train)
print(5)
'''
