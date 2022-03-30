import argparse
from ConvLSTM import *

parser = argparse.ArgumentParser(
    description="Training ConvLSTM model (Keras/TensorFlow)")

parser.add_argument('-e', '--epoch', default=100,
                    help='number of epoch')

parser.add_argument('-m', '--model_name', default='frame',
                    help='model_name structure')

parser.add_argument('-n', '--name', default='917/ConvLSTM',
                    help='name of the model')

parser.add_argument('-i', '--input_length', default=64,
                    help='input_length ')

parser.add_argument('-f', '--frames', default=1,
                    help='number of input frames')

parser.add_argument('--label', default=16,
                    help='label_length')

parser.add_argument('--numeric', type=bool, default=False,
                    help='numeric problem')

parser.add_argument('--VM', default=917,
                    help='VM number')


args = parser.parse_args()


EPOCH = int(args.epoch)
MODEL_NAME = args.model_name
NAME = args.name
INPUT_LENGTH = int(args.input_length)
LABEL_LENGTH = int(args.label)
FRAMES = int(args.frames)
NUMERIC = bool(args.numeric)
VM_NUM = int(args.VM)

if __name__ == "__main__":
    print('Pre-processing')
    # Load data
    VM = load_VM(f'{VM_NUM}.csv')
    # VM = load_VM('541.csv')
    # Make it univariate
    df = VM[['CPU usage [MHZ]']]

    # Split the data
    # (70%, 20%, 10%) split for the training, validation, and test sets
    train_df, val_df, test_df = split_data(df)

    # Normalizing
    scaler = MinMaxScaler()
    train_df, val_df, test_df = data_transformation(scaler, train_df, val_df, test_df)

    # Model
    ConvLSTM_model = ConvLSTMModel(input_width=INPUT_LENGTH,
                                   label_width=LABEL_LENGTH,
                                   n_frames=FRAMES,
                                   df=df,
                                   model_name=MODEL_NAME,
                                   name=NAME,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   epoch=EPOCH,
                                   model_path=None,
                                   numeric=NUMERIC,
                                   )

    # Training
    print('Training:')
    history = ConvLSTM_model.compile_and_fit()
    # Prediction
    print('Prediction:')
    if ConvLSTM_model.numeric is False:
        pred, img_pred, pred_df_trf = ConvLSTM_model.prediction(scaler)
    else:
        pred, pred_df_trf = ConvLSTM_model.prediction(scaler)
    # Evaluation
    print('Evaluation:')
    metrics = ConvLSTM_model.evaluation(pred_df_trf, scaler)
