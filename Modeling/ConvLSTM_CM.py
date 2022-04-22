import argparse
from ConvLSTM import *

parser = argparse.ArgumentParser(
    description="Training ConvLSTM model (Keras/TensorFlow)  (Cross-Modeling)")

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

parser.add_argument('--VM_test', default=340,
                    help='VM number')

parser.add_argument('--overlapping', default=None,
                    help='Overlapping (0-1)')

parser.add_argument('--img_width', default=None,
                    help='Image width (multiple of input_width)')


args = parser.parse_args()


EPOCH = int(args.epoch)
MODEL_NAME = args.model_name
NAME = args.name
INPUT_LENGTH = int(args.input_length)
LABEL_LENGTH = int(args.label)
FRAMES = int(args.frames)
NUMERIC = bool(args.numeric)
VM_NUM = int(args.VM)
VM_NUM_test = int(args.VM_test)
OVERLAPPING = args.overlapping
IMAGE_WIDTH = args.img_width

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

    # Model
    ConvLSTM_model = ConvLSTMModel(input_width=INPUT_LENGTH,
                                   label_width=LABEL_LENGTH,
                                   n_frames=FRAMES,
                                   df=df,
                                   image_width=IMAGE_WIDTH,
                                   model_name=MODEL_NAME,
                                   name=NAME,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   epoch=EPOCH,
                                   model_path=None,
                                   numeric=NUMERIC,
                                   overlapping=OVERLAPPING,
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
