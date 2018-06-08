from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import neural_network as nn
import os

model_names = ["simple_model","three_layer_model"]
epochs = [100,100]
batch_sizes = [32,256]
losses = ["categorical_crossentropy", "mean_squared_error","mean_absolute_error"]
optimizers = ["sgd", "adam"]

log_folder = "log"
log_file = os.path.join("log","log_nn.txt")

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
    def to_str():
        return ("%s,%s,%s,%s,%s"%(name,epochs,batch_size,loss,optimizer))
    #alternative structure
    def get_score():
        self.score = nn.get_score(self)
    def __lt__(params):
        return self.score[1] > params.score[1]
    def __eq__(params):
        return self.score[1] == params.score[1]

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

def split_interval(interval,split_count=10):
    dist = (interval[1]-interval[0]) / split_count
    last_value = interval[0]
    for i in range(1+split_count):
        yield last_value
        last_value += dist

def setup_log():

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Log Neural Network\n")
            f.write("[Loss,Acc], batch_size, epoch\n")

def write_log(score, batch_size, epoch):
    with open(log_file, 'a') as f:
        f.write("%s,%s,%s\n"%(score, int(batch_size), epoch))

if __name__ == "__main__":
    setup_log()
    scores = []
    params = get_default()
    for batch_size in split_interval(batch_sizes,20):
        params.batch_size = int(batch_size)
        score = nn.get_score(params)
        write_log(score, batch_size, epochs)
        scores.append((score,params))
