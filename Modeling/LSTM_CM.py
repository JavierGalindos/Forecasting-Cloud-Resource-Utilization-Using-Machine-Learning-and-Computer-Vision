import argparse
from LSTM import *

parser = argparse.ArgumentParser(
    description="Training LSTM model (Keras/TensorFlow)")

parser.add_argument('-e', '--epoch', default=100,
                    help='number of epoch')

parser.add_argument('-n', '--name', default='599/LSTM',
                    help='name of the model')

parser.add_argument('-s', '--figure_path', default="599",
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

parser.add_argument('--classification', type=bool, default=False,
                    help='classification problem')

parser.add_argument('--VM_test', default=340,
                    help='VM number')

parser.add_argument('--VM', default=917,
                    help='VM number')

args = parser.parse_args()


EPOCH = int(args.epoch)
NAME = args.name
INPUT_LENGTH = int(args.input_length)
LABEL_LENGTH = int(args.label)
HIDDEN_DIM = int(args.dim)
N_LAYERS = int(args.layers)
DROPOUT = float(args.dropout)
CLASSIFICATION = bool(args.classification)
figure_path = args.figure_path
VM_NUM = int(args.VM)
VM_NUM_test = int(args.VM_test)

# Check figure path
figures_path = os.path.join(FIGURES_PATH, figure_path)
if not os.access(figures_path, os.F_OK):
    os.makedirs(figures_path)
if not os.access(figures_path, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(figures_path))
    exit()
else:
    print('figures saved to {}'.format(figures_path))

FIGURES_PATH = figures_path

if __name__ == "__main__":
    print('Pre-processing')
    # Load data
    VM = load_VM(f'{VM_NUM}.csv')
    # Make it univariate
    df = VM[['CPU usage [MHZ]']]

    # Make data sythetic
    # df = synthetic_dataset(df, 1/25)

    # Split the data
    # (80%, 20%, 0%) split for the training, validation from one VM
    train_df, val_df, _ = split_data(df, 0.8, 0.19)
    # Test set from other VM
    VM_test = load_VM(f'{VM_NUM_test}.csv')
    df_test = VM_test[['CPU usage [MHZ]']]

    # Normalizing
    # Train & Validation
    scaler = MinMaxScaler()
    train_df, val_df, _ = data_transformation(scaler, train_df, val_df, _)
    # Test
    scaler = MinMaxScaler()
    df_test.loc[:, df_test.columns] = scaler.fit_transform(df_test.loc[:, df_test.columns])
    test_df = df_test.copy()

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
