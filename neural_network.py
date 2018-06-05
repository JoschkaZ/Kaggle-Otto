import pandas as pd
from helper_functions import *
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import os

#DATA SETTINGS
label = 'target'
no_feature_columns = [label, 'id']
cat_to_int_columns = []
cat_to_onehot_columns = ['target']
K = 5

data = Data([label], [label], K, no_feature_columns)
data.col_to_onehot(label)
#CLASSIFIER SETTINGS
epochs = 10
batch_size = 64
model_folder = "models"
evaluation_file = os.path.join(model_folder,"evaluations.txt")

#
def build_simple_model(in_dim, out_dim):
    model = Sequential()
    model.add(Dense(186, activation="relu", input_dim = in_dim))
    model.add(Dense(out_dim, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model, "simple_model"

def build_deep_model():
    return None

model, model_name = build_simple_model(data.get_feature_shape(), data.get_label_shape())

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

log = "%s ; [loss, acc] : %s ; epochs : %s ; batch size : %s ; folds : %s\n" %(model_name, str(loss_and_metrics), epochs, batch_size, data.fold_count)
with open(evaluation_file, 'a') as f:
    f.write(log)

model.save(os.path.join(model_folder, "%s_%s.h5" % (model_name,epochs)))
