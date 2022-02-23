import time_series  # custom TS methods
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import pandas as pd
import os

FIGURES_PATH = '../../Figures/Modeling/LSTM'


class Data_Prep:

    def __init__(self, dataset):
        self.dataset = dataset

    def preprocess_rnn(self, date_colname, numeric_colname, pred_set_timesteps):
        features = (time_series.create_series(self.dataset, numeric_colname, date_colname)).sort_index()
        rnn_df = features.groupby(features.index).sum()

        # Filter out 'n' timesteps for prediction purposes
        timestep_idx = len(rnn_df) - pred_set_timesteps
        validation_df = rnn_df.iloc[timestep_idx:]
        rnn_df = rnn_df.iloc[1:timestep_idx, ]

        # Dickey Fuller Test
        print("Summary Statistics - ADF Test For Stationarity\n")
        if time_series.stationarity_test(X=rnn_df[numeric_colname], return_p=True, print_res=False) > 0.05:
            print("P Value is high. Consider Differencing: " + str(
                time_series.stationarity_test(X=rnn_df[numeric_colname], return_p=True, print_res=False)))
        else:
            time_series.stationarity_test(X=rnn_df[numeric_colname])

        # Sorting
        rnn_df = rnn_df.sort_index(ascending=True)
        rnn_df = rnn_df.reset_index()

        return rnn_df, validation_df


class Series_Prep:

    def __init__(self, rnn_df, numeric_colname):
        self.rnn_df = rnn_df
        self.numeric_colname = numeric_colname

    def make_window(self, sequence_length, train_test_split, return_original_x=True):

        # Create the initial results df with a look_back of 60 days
        result = []

        # 3D Array
        for index in range(len(self.rnn_df) - sequence_length):
            result.append(self.rnn_df[self.numeric_colname][index: index + sequence_length])

        # Getting the initial train_test split for our min/max val scalar
        train_test_split = 0.9
        row = int(round(train_test_split * np.array(result).shape[0]))
        train = np.array(result)[:row, :]
        X_train = train[:, :-1]

        # Manual MinMax Scaler
        X_min = X_train.min()
        X_max = X_train.max()

        # keep the originals in case
        X_min_orig = X_train.min()
        X_max_orig = X_train.max()

        # Minmax scaler and a reverse method
        def minmax(X):
            return (X - X_min) / (X_max - X_min)

        def reverse_minmax(X):
            return X * (X_max - X_min) + X_min

        # Method for Scaler for each window in our 3D array
        def minmax_windows(window_data):
            normalised_data = []
            for window in window_data:
                window.index = range(sequence_length)
                normalised_window = [((minmax(p))) for p in window]
                normalised_data.append(normalised_window)
            return normalised_data

        # minmax the windows
        result = minmax_windows(result)
        # Convert to 2D array
        result = np.array(result)
        if return_original_x:
            return result, X_min_orig, X_max_orig
        else:
            return result

    @staticmethod
    def reshape_window(window, train_test_split=0.8):
        # Train/test for real this time
        row = round(train_test_split * window.shape[0])
        train = window[:row, :]

        # Get the sets
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = window[row:, :-1]
        y_test = window[row:, -1]

        # Reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))

        return X_train, X_test, y_train, y_test


class Predict_Future:

    def __init__(self, X_test, validation_df, lstm_model):
        self.X_test = X_test
        self.validation_df = validation_df
        self.lstm_model = lstm_model

    def predicted_vs_actual(self, X_min, X_max, numeric_colname):

        curr_frame = self.X_test[len(self.X_test) - 1]
        future = []

        for i in range(len(self.validation_df)):
            # append the prediction to our empty future list
            future.append(self.lstm_model.predict(curr_frame[newaxis, :, :])[0, 0])
            # insert our predicted point to our current frame
            curr_frame = np.insert(curr_frame, len(self.X_test[0]), future[-1], axis=0)
            # push the frame up one to make it progress into the future
            curr_frame = curr_frame[1:]

        def reverse_minmax(X, X_max=X_max, X_min=X_min):
            return X * (X_max - X_min) + X_min

        # Plot
        reverse_curr_frame = pd.DataFrame(
            {numeric_colname: [reverse_minmax(x) for x in self.X_test[len(self.X_test) - 1]],
             "historical_flag": 1})
        reverse_future = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in future],
                                       "historical_flag": 0})

        # Change the indicies! Only for FUTURE predictions
        # reverse_future.index += len(reverse_curr_frame)

        print("See Plot for predicted vs. actuals")
        fig = plt.figure()
        plt.plot(reverse_curr_frame[numeric_colname])
        plt.plot(reverse_future[numeric_colname])
        plt.title("Predicted Points Vs. Actuals (Validation)")
        # plt.show()
        save_path = os.path.join(FIGURES_PATH, 'predicted_actual')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Check accuracy vs. actuals
        comparison_df = pd.DataFrame({"Validation": self.validation_df[numeric_colname],
                                      "Predicted": [reverse_minmax(x) for x in future]})
        print("Validation Vs. Predicted")
        print(comparison_df.sum())

    def predict_future(self, X_min, X_max, numeric_colname, timesteps_to_predict, return_future=True):

        curr_frame = self.X_test[len(self.X_test) - 1]
        future = []

        for i in range(timesteps_to_predict):
            # append the prediction to our empty future list
            future.append(self.lstm_model.predict(curr_frame[newaxis, :, :])[0, 0])
            # insert our predicted point to our current frame
            curr_frame = np.insert(curr_frame, len(self.X_test[0]), future[-1], axis=0)
            # push the frame up one to make it progress into the future
            curr_frame = curr_frame[1:]

        def reverse_minmax(X, X_max=X_max, X_min=X_min):
            return X * (X_max - X_min) + X_min

        # Reverse the original frame and the future frame
        reverse_curr_frame = pd.DataFrame(
            {numeric_colname: [reverse_minmax(x) for x in self.X_test[len(self.X_test) - 1]],
             "historical_flag": 1})
        reverse_future = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in future],
                                       "historical_flag": 0})

        # Change the indicies to show prediction next to the actuals in orange
        reverse_future.index += len(reverse_curr_frame)

        print("See Plot for Future Predictions")
        fig = plt.figure()
        plt.plot(reverse_curr_frame[numeric_colname])
        plt.plot(reverse_future[numeric_colname])
        plt.title("Predicted Future of " + str(timesteps_to_predict) + " days")
        # plt.show()
        save_path = os.path.join(FIGURES_PATH, 'future_predictions')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        if return_future:
            return reverse_future