import numpy as np
import pandas as pd
from PreProcess import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

from paths import PATH

#SETTINGS
LABEL = 'target'
NOT_A_FEATURE = [LABEL, 'id']
K = 5
CAT_TO_INT_COLS = ['target'] #categorical columns to be transformed into integers

#READ DATA
train=pd.read_csv(PATH + r'train.csv')
test=pd.read_csv(PATH + r'test.csv')

#GET FEATURES
features = list(train.columns.values)
features = [x for x in features if x not in NOT_A_FEATURE]

#DATA TRANSFORMATIONS
train = cat_to_int(train, CAT_TO_INT_COLS)
train = add_kfold_indexes(train, K)

#CLASSIFIER SETTINGS
rd= RandomForestClassifier(n_estimators=100)

#LOOP OVER FOLDS
for k in range(K):
    print('Using Fold: ', k)
    x_train_train, y_train_train, x_train_test, y_train_test = kfold_split(train, k, LABEL, features)
    rd.fit(x_train_train, y_train_train)
    pred = rd.predict(x_train_test)
    print(accuracy_score(pred, y_train_test))
