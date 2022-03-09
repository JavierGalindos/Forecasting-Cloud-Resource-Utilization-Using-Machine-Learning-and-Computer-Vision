from LSTM import *
from sklearn.model_selection import TimeSeriesSplit

# Hyperparameters
EPOCH = 100
NAME = 'LSTM_CV'
INPUT_LENGTH = 10
LABEL_LENGTH = 1
HIDDEN_DIM = 20
N_LAYERS = 1
DROPOUT = 0
CLASSIFICATION = False

