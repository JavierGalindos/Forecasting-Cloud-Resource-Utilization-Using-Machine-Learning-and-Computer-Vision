from ConvLSTM import *
from sklearn.model_selection import TimeSeriesSplit

# Hyperparameters
EPOCH = 100
MODEL_NAME = 'video'
NAME = '917/CV'
INPUT_LENGTH = 64
LABEL_LENGTH = 8
FRAMES = 16
NUMERIC = False
VM_NUM = 917