import argparse

import pandas as pd

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

# parser.add_argument('--VM_test', default=340,
#                     help='VM number')

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
# VM_NUM_test = int(args.VM_test)
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
    VM_NUM_test = 1
    VM_test = load_VM(f'{VM_NUM_test}.csv')
    df_test = VM_test[['CPU usage [MHZ]']]

    # Normalizing
    # Train & Validation
    scaler = MinMaxScaler()
    train_df, val_df, _ = data_transformation(scaler, train_df, val_df, _)

    # More than one VM as training set
    VM_NUM = 3  # TODO: remove after training
    # Step
    VM = load_VM(f'{226}.csv')
    # Make it univariate
    new_train = VM[['CPU usage [MHZ]']]
    # Scale
    scaler = MinMaxScaler()
    new_train, _, _ = data_transformation(scaler, new_train, _, _)
    train_df = pd.concat([train_df, new_train], ignore_index=True)
    # Random
    VM = load_VM(f'{226}.csv')
    # Make it univariate
    new_train = VM[['CPU usage [MHZ]']]
    scaler = MinMaxScaler()
    new_train, _, _ = data_transformation(scaler, new_train, _, _)
    train_df = pd.concat([train_df, new_train], ignore_index=True)

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
    # Prediction for every VM
    for VM in range(1, 1250):
        VM_test = load_VM(f'{VM}.csv')
        df_test = VM_test[['CPU usage [MHZ]']]
        # Test
        scaler = MinMaxScaler()
        df_test.loc[:, df_test.columns] = scaler.fit_transform(df_test.loc[:, df_test.columns])
        test_df = df_test.copy()

        # Change class attributes
        setattr(ConvLSTM_model, 'test_df', test_df)
        setattr(ConvLSTM_model, 'name', f'917/CM_{VM_NUM}/{VM}')
        # Change test_pred too
        test_pred_df = pd.concat(
            [ConvLSTM_model.val_df.iloc[
             -(ConvLSTM_model.input_width + ConvLSTM_model.label_width * (ConvLSTM_model.n_frames - 1)):, :],
             ConvLSTM_model.test_df])
        setattr(ConvLSTM_model, 'test_pred_df', test_pred_df)

        print('Prediction:')
        try:
            if ConvLSTM_model.numeric is False:
                pred, img_pred, pred_df_trf = ConvLSTM_model.prediction(scaler)
            else:
                pred, pred_df_trf = ConvLSTM_model.prediction(scaler)
            # Evaluation
            print('Evaluation:')
            metrics = ConvLSTM_model.evaluation(pred_df_trf, scaler)
        except:
            print(f"VM{VM} failed")