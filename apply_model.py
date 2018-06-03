import nn_preprocess as preprocess
from keras.models import load_model
import sys
import numpy as np

#for example filename = "simple_model_50.h5"
def predict(filename, data = preprocess.get_test()):
    model = load_model(filename)
    return model.predict(data)



if __name__=="__main__":
    filename = "model.h5"
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print("standard filename model.h5 used")
    prediction = predict(filename)
    prediction = preprocess.one_hot_inverse(prediction)
    np.savetxt("prediction.csv", prediction, delimiter=",")
