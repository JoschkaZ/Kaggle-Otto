import pandas as pd
from helper_functions import *
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import os
import neural_models
from neural_models import model_names
from neural_models import ModelParams

#DATA SETTINGS
label = 'target'
no_feature_columns = [label, 'id']
cat_to_int_columns = ['target']
cat_to_onehot_columns = ['target']
K = 5

data = Data([label], [label],to_onehot_columns=[label], fold_count = K, no_feature_columns = no_feature_columns)
#data.col_to_onehot(label)

model_folder = "models"
evaluation_file = os.path.join(model_folder,"evaluations.txt")

def get_score(params : ModelParams, save = False, log = False, overfit = False):
    model, model_name = neural_models.get_model(params.name, data.get_feature_shape(), data.get_label_shape())
    model.compile(loss = params.loss, optimizer = params.optimizer, metrics=["accuracy"])

    if overfit:
        for k in range(data.fold_count):
            print('Using Fold: ', k)
            x_train_train, y_train_train, x_train_test, y_train_test = data.fold_split(k)
            model.fit(x_train_train, y_train_train, epochs=params.epochs, batch_size=params.batch_size, verbose=params.verbose)
    else:
        k = 0
        x_train_train, y_train_train, x_train_test, y_train_test = data.fold_split(k)
        model.fit(x_train_train, y_train_train, epochs=params.epochs, batch_size=params.batch_size, verbose=params.verbose)

    loss_and_metrics = model.evaluate(x_train_test, y_train_test, batch_size=128, verbose=0)

    #create log and save
    #save==True -> log
    if log or save:

        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        if not os.path.exists(evaluation_file):
            with open(evaluation_file, 'w') as f:
                f.write("Evaluations\n")

        log_text = "%s ; [loss, acc] : %s ; epochs : %s ; batch size : %s ; folds : %s ; loss : %s ; optimizer : %s\n"
                        %(params.name, str(loss_and_metrics), params.epochs, params.batch_size, data.fold_count, params.loss, params.optimizer)
        with open(evaluation_file, 'a') as f:
            f.write(log_text)
        if save:
            model.save(os.path.join(model_folder, "%s_%s.h5" % (params.name,params.epochs)))
    return loss_and_metrics

#deprecated below
#alternative usage

epochs = 20
batch_size = 32
loss = "categorical_crossentropy"
optimizer = "sgd"

def build_simple_model(in_dim, out_dim, loss = loss, optimizer = optimizer):
    model = Sequential()
    model.add(Dense(186, activation="relu", input_dim = in_dim))
    model.add(Dense(out_dim, activation="softmax"))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model, "simple_model"

def build_deep_model(in_dim, out_dim, loss = loss, optimizer=optimizer):
    model = Sequential()
    model.add(Dense(186, activation="relu", input_dim=in_dim))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(out_dim, activation="softmax"))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model, "three_layer_model"

if __name__ == "__main__":

    model, model_name = build_deep_model(data.get_feature_shape(), data.get_label_shape())

    for k in range(K):
        print('Using Fold: ', k)
        x_train_train, y_train_train, x_train_test, y_train_test = data.fold_split(k)
        model.fit(x_train_train, y_train_train, epochs=epochs, batch_size=batch_size, verbose=0)

    loss_and_metrics = model.evaluate(x_train_test, y_train_test, batch_size=128, verbose=0)
    print(loss_and_metrics)

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    if not os.path.exists(evaluation_file):
        with open(evaluation_file, 'w') as f:
            f.write("Evaluations\n")

    log = "%s ; [loss, acc] : %s ; epochs : %s ; batch size : %s ; folds : %s ; loss : %s ; optimizer : %s\n" %(model_name, str(loss_and_metrics), epochs, batch_size, data.fold_count, loss, optimizer)
    with open(evaluation_file, 'a') as f:
        f.write(log)

    model.save(os.path.join(model_folder, "%s_%s.h5" % (model_name,epochs)))
