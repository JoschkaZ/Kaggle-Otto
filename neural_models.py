from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

model_names = ["simple_model","three_layer_model"]
epochs = [20,20]
batch_sizes = [32,256]
losses = ["categorical_crossentropy", "mean_squared_error","mean_absolute_error"]
optimizers = ["sgd", "adam"]

param = {"names": model_names,
            "epochs":epochs,
            "batch_size":batch_sizes,
            "losses": losses,
            "optimizers":optimizers}

class ModelParams:
    def __init__(self, name, epochs, batch_size, loss, optimizer, verbose = 0):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose

def get_default():
    return ModelParams(model_names[0], epochs[0], batch_sizes[0], losses[0], optimizers[0])

def build_simple_model(in_dim, out_dim, model):
    model.add(Dense(186, activation="relu", input_dim = in_dim))
    model.add(Dense(out_dim, activation="softmax"))
    return model, "simple_model"

def build_deep_model(in_dim, out_dim, model):
    model.add(Dense(186, activation="relu", input_dim=in_dim))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(out_dim, activation="softmax"))
    return model, "three_layer_model"

def get_model(name, in_dim, out_dim):
    model = Sequential()
    if name == model_names[0]:
        return build_simple_model(in_dim, out_dim, model)
    if name == model_names[1]:
        return build_deep_model(in_dim, out_dim, model)
