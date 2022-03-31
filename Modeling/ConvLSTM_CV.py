from ConvLSTM import *
from sklearn.model_selection import TimeSeriesSplit

# Hyperparameters
EPOCH = 100
MODEL_NAME = 'video'
NAME = '917/CV'
INPUT_LENGTH = 64
LABEL_LENGTH = 8
FRAMES = 16
NUMERIC = False
VM_NUM = 917

if __name__ == "__main__":
    # Load data
    VM = load_VM(f'{VM_NUM}.csv')
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

        metrics_cv.append(metrics)

        # Saving metrics
        metrics_cv_all = pd.concat(metrics_cv, axis=1)
        metrics_cv_all = metrics_cv_all.T.reset_index()
        # Save in file
        try:
            filename = os.path.join('logs', ConvLSTM_model.name, 'metrics_cv.csv')
            metrics_cv_all.to_csv(filename)
        except:
            print("Unable to write to file")

