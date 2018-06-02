import numpy as np
import pandas as pd
from PreProcess import *
from sklearn.ensemble import RandomForestClassifier

PATH = r'C:\Data\Otto'
LABEL = 'TARGET'



train=pd.read_csv(PATH + r'\train.csv')
test=pd.read_csv(PATH + r'\test.csv')


train = add_kfold_indexes(5,train)

print(train)




rd= RandomForestClassifier(n_estimators=100)

#rd.fit(X_train,y_train)
