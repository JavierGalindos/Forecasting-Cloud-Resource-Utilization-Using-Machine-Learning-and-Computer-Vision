from LSTM import *
from sklearn.model_selection import TimeSeriesSplit

# Hyperparameters
EPOCH = 100
NAME = '549/CV6'
INPUT_LENGTH = 1
LABEL_LENGTH = 6
HIDDEN_DIM = 20
N_LAYERS = 1
DROPOUT = 0
CLASSIFICATION = False

if __name__ == "__main__":
    # Load data
    # VM = load_VM('541.csv')
    VM = load_VM('599.csv')
    # Make it univariate
    df = VM[['CPU usage [MHZ]']]

    # Cross-validation split
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_cv = []

    for fold, idx in enumerate(tscv.split(df)):
        print(f'Fold {fold}:')
        # Get indexes for each fold (incremental for time series)
        idxs = np.concatenate((idx[0], idx[1]))
        df_fold = df.iloc[idxs, :]
        NAME_FOLD = f'{NAME}_{fold}'

        # Data pipeline
        # Split the data
        # (70%, 20%, 10%) split for the training, validation, and test sets
        train_df, val_df, test_df = split_data(df_fold)

        # Normalizing
        scaler = MinMaxScaler()
        train_df, val_df, test_df = data_transformation(scaler, train_df, val_df, test_df)

        # LSTM model
        lstm_model = LstmModel(input_width=INPUT_LENGTH, label_width=LABEL_LENGTH, df=df_fold, train_df=train_df,
                               val_df=val_df,
                               test_df=test_df, epoch=EPOCH, units=HIDDEN_DIM, layers=N_LAYERS, dropout=DROPOUT,
                               name=NAME_FOLD,
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
        metrics_cv.append(metrics)

        # Saving metrics
        metrics_cv_all = pd.concat(metrics_cv, axis=1)
        metrics_cv_all = metrics_cv_all.T.reset_index()
        # Save in file
        try:
            filename = os.path.join('logs', lstm_model.name, 'metrics_cv.txt')
            metrics_cv_all.to_csv(filename)
        except:
            print("Unable to write to file")
