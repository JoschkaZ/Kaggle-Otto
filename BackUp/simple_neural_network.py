import numpy as np
import nn_preprocess as preprocess
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import sys

def build_simple_model(in_dim, out_dim):
    model = Sequential()
    model.add(Dense(186, activation="relu", input_dim = in_dim))
    model.add(Dense(out_dim, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    epochs = 100
    if len(sys.argv) == 2:
        epochs = int(sys.argv[1])
    x_train, y_train = preprocess.get_train()
    model = build_simple_model(preprocess.FEATURE_COUNT, preprocess.CLASS_COUNT)
    model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
    model.save("simple_model_%s.h5" % (epochs))
    print(loss_and_metrics)
