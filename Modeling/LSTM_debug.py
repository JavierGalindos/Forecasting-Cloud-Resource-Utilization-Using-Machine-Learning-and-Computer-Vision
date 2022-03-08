from LSTM import *

if __name__ == "__main__":
    print('Pre-processing')
    # Load data
    VM = load_VM('917.csv')
    # Make it univariate
    df = VM[['CPU usage [MHZ]']]

    # Split the data
    # (70%, 20%, 10%) split for the training, validation, and test sets
    train_df, val_df, test_df = split_data(df)

    # Normalizing
    scaler = MinMaxScaler()
    train_df, val_df, test_df = data_transformation(scaler, train_df, val_df, test_df)

    # LSTM model
    lstm_model = LstmModel(input_width=10,
                           label_width=3,
                           name='LSTM',
                           df=df,
                           train_df=train_df,
                           val_df=val_df,
                           test_df=test_df,
                           epoch=100,
                           units=20,
                           layers=1,
                           dropout=0,
                           )

    train = lstm_model.window.train
    # Training
    print('Training:')
    history = lstm_model.compile_and_fit(patience=50)
    # Prediction
    print('Prediction:')
    pred = lstm_model.prediction(scaler)
    # Evaluation
    print('Evaluation:')
    metrics = lstm_model.evaluation(pred, scaler)