from ConvLSTM import *

if __name__ == '__main__':
    print('Pre-processing')
    # Load data
    VM_NUM = 917
    VM = load_VM('917.csv')
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
    ConvLSTM_model = ConvLSTMModel(input_width=288,
                                   label_width=16,
                                   df=df,
                                   image_width=64,
                                   model_name="AE",
                                   name='917/input_288_img_64',
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   epoch=1,
                                   batch_size=16,
                                   # model_path='./logs/ConvLSTM/917/frames_8/checkpoints/ConvLSTM_model_2022_03_25__16_50_02.hdf5',
                                   n_frames=1,
                                   )

    # Training
    print('Training:')
    history = ConvLSTM_model.compile_and_fit()
    # Prediction
    print('Prediction:')
    pred, img_pred, pred_df_trf = ConvLSTM_model.prediction(scaler)
    # Evaluation
    metrics = ConvLSTM_model.evaluation(pred_df_trf, scaler)

