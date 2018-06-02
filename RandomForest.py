import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from PreProcess import *

PATH = r'C:\Data\Otto'



train=pd.read_csv(PATH + r'\train.csv')
test=pd.read_csv(PATH + r'\test.csv')


train = add_kfold_indexes(5,train)

print(train)
