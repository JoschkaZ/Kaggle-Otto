import pandas as pd
from helper_functions import *
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

#DATA SETTINGS
label = 'target'
no_feature_columns = [label, 'id']
cat_to_int_columns = []
cat_to_onehot_columns = ['target']
K = 5

train, test, features, labels = get_data(label = label,
                                no_feature_columns = no_feature_columns,
                                cat_to_int_columns = cat_to_int_columns,
                                cat_to_onehot_columns = cat_to_onehot_columns,
                                K = K)

#CLASSIFIER SETTINGS
epochs = 10
batch_size = 200
model = Sequential()
model.add(Dense(186, activation="relu", input_dim = len(features)))
model.add(Dense(len(labels), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

for k in range(K):
    print('Using Fold: ', k)
    x_train_train, y_train_train, x_train_test, y_train_test = kfold_split(train, k, labels, features)

    model.fit(x_train_train, y_train_train, epochs=epochs, batch_size=32, verbose=0)

    loss_and_metrics = model.evaluate(x_train_test, y_train_test, batch_size=128, verbose=0)
    print(loss_and_metrics)

    #model.save("simple_model_%s.h5" % (epochs))
