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
    ConvLSTM_model = ConvLSTMModel(input_width=64,
                                   label_width=16,
                                   df=df,
                                   name='ConvLSTM',
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   epoch=50,
                                   batch_size=16,
                                   model_path=None,
                                   )

    # Training
    print('Training:')
    history = ConvLSTM_model.compile_and_fit()
    # Prediction
    print('Prediction:')
    pred, img_pred, pred_df_trf = ConvLSTM_model.prediction(scaler)
