from paths import PATH
from pre_process_functions import *

def get_stuff():
    1

#SETTINGS
LABEL = 'target'
NOT_A_FEATURE = [LABEL, 'id']

#GET FEATURES
features = list(train.columns.values)
features = [x for x in features if x not in NOT_A_FEATURE]

#READ DATA
train=pd.read_csv(PATH + r'train.csv')
test=pd.read_csv(PATH + r'test.csv')

train_test = train_test_merge(train,test, LABEL)



print(train_test)
