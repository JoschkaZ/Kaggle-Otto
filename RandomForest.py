import numpy as np
import pandas as pd
from PreProcess import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

#SETTINGS
PATH = r'C:\Data\Otto'
LABEL = 'target'
K = 5
CAT_TO_INT_COLS = ['target'] #categorical columns to be transformed into integers

train=pd.read_csv(PATH + r'\train.csv')
test=pd.read_csv(PATH + r'\test.csv')
train_test = train_test_merge(train,test, LABEL)
train_test = cat_to_int(train_test, CAT_TO_INT_COLS)
train, test = train_test_split(train_test)
train = add_kfold_indexes(K,train)

train.to_csv(r"C:\Users\Joschka\Desktop\train.csv")

'''
rd= RandomForestClassifier(n_estimators=100)

for k in range(K):
    x_train_train, y_train_train, x_train_test, y_train_test = kfold_split(train, k, LABEL)

    display(y_train.head(5))
    input("...")
    rd.fit(x_train,y_train)
print(5)
'''
