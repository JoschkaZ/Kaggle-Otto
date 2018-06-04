import pandas as pd
from helper_functions import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

#DATA SETTINGS
label = 'target'
no_feature_columns = [label, 'id']
cat_to_int_columns = ['target']
cat_to_onehot_columns = []
K = 5

train, test, features, labels = get_data(label = label,
                                no_feature_columns = no_feature_columns,
                                cat_to_int_columns = cat_to_int_columns,
                                cat_to_onehot_columns = cat_to_onehot_columns,
                                K = K)

#CLASSIFIER SETTINGS
rd= RandomForestClassifier(n_estimators=5)

for k in range(K):
    print('Using Fold: ', k)
    x_train_train, y_train_train, x_train_test, y_train_test = kfold_split(train, k, label, features)

    rd.fit(x_train_train, y_train_train)

    pred = rd.predict(x_train_test)
    print(accuracy_score(pred, y_train_test))
