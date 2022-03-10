import argparse
from LSTM import *

parser = argparse.ArgumentParser(
    description="Training LSTM model (Keras/TensorFlow)")

parser.add_argument('-e', '--epoch', default=100,
                    help='number of epoch')

parser.add_argument('-n', '--name', default='541/LSTM',
                    help='name of the model')

parser.add_argument('-s', '--figure_path', default="541",
                    help='Destination for figures to be saved at')

parser.add_argument('-i', '--input_length', default=50,
                    help='input_length ')

parser.add_argument('-d', '--dim', default=20,
                    help='number of neurons per layer')

parser.add_argument('-l', '--layers', default=1,
                    help='number of rnn layers')

parser.add_argument('--dropout', default=0,
                    help='dropout')

parser.add_argument('--label', default=1,
                    help='label_length')

parser.add_argument('--classification', default=False,
                    help='classification problem')

args = parser.parse_args()

EPOCH = int(args.epoch)
NAME = args.name
INPUT_LENGTH = int(args.input_length)
LABEL_LENGTH = int(args.label)
HIDDEN_DIM = int(args.dim)
N_LAYERS = int(args.layers)
DROPOUT = float(args.dropout)
CLASSIFICATION = args.classification
figure_path = args.figure_path

# Check figure path
figures_path = os.path.join(FIGURES_PATH, figure_path)
if not os.access(figures_path, os.F_OK):
    os.mkdir(figures_path)
if not os.access(figures_path, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(figures_path))
    exit()
else:
    print('figures saved to {}'.format(figures_path))

FIGURES_PATH = figures_path

if __name__ == "__main__":
    print('Pre-processing')
    # Load data
    VM = load_VM('541.csv') # 917.csv
    # Make it univariate
    df = VM[['CPU usage [MHZ]']]

    # Split the data
    # (70%, 20%, 10%) split for the training, validation, and test sets
    train_df, val_df, test_df = split_data(df)

    # Normalizing
    scaler = MinMaxScaler()
    train_df, val_df, test_df = data_transformation(scaler, train_df, val_df, test_df)

    # LSTM model
    lstm_model = LstmModel(input_width=INPUT_LENGTH, label_width=LABEL_LENGTH, df=df, train_df=train_df, val_df=val_df,
                           test_df=test_df, epoch=EPOCH, units=HIDDEN_DIM, layers=N_LAYERS, dropout=DROPOUT, name=NAME,
                           classification=CLASSIFICATION)

    # Training
    print('Training:')
    history = lstm_model.compile_and_fit(patience=80)
    # Prediction
    print('Prediction:')
    pred = lstm_model.prediction(scaler)
    # Evaluation
    print('Evaluation:')
    metrics = lstm_model.evaluation(pred, scaler)
