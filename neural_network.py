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

data = Data([label], [label], K, no_feature_columns)
#data.col_to_onehot(label, in_place= True)
#CLASSIFIER SETTINGS
epochs = 10
batch_size = 200

def build_simple_model():
    model = Sequential()
    model.add(Dense(186, activation="relu", input_dim = len(data.features)))
    model.add(Dense(len(labels), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model

model = build_simple_model()

for k in range(K):
    print('Using Fold: ', k)
    x_train_train, y_train_train, x_train_test, y_train_test = data.fold_split(k) #kfold_split(train, k, labels, features)
    y_train_train = data.to_onehot(y_train_train[:,0]) #
    y_train_test = data.to_onehot(y_train_test[:,0])
    model.fit(x_train_train, y_train_train, epochs=epochs, batch_size=32, verbose=1)

    loss_and_metrics = model.evaluate(x_train_test, y_train_test, batch_size=128, verbose=0)
    print(loss_and_metrics)

    #model.save("simple_model_%s.h5" % (epochs))
