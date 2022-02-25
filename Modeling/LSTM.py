import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime

from DataExploration.BitbrainsUtils import load_VM, plot_timeSeries

FIGURES_PATH = '../Figures/Modeling/LSTM'

if not os.access(FIGURES_PATH, os.F_OK):
    os.mkdir(FIGURES_PATH)
if not os.access(FIGURES_PATH, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(FIGURES_PATH))
    exit()
else:
    print('figures saved to {}'.format(FIGURES_PATH))


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Dataframe for predictions (take test + input_length from validation set)
        self.test_pred_df = pd.concat([self.val_df.iloc[-self.input_width:, :], self.test_df])

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='CPU usage [MHZ]', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def test_pred(self):
        return self.make_dataset(self.test_pred_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def add_daily_info(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_s = df.index.map(pd.Timestamp.timestamp)

    day = 60 * 24 / 5
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    return df


def split_data(df: pd.DataFrame, training: float = 0.7, validation: float = 0.2, test: float = 0.1):
    n = len(df)
    train_df = df[0:int(n * training)]
    val_df = df[int(n * training):int(n * (training + validation))]
    test_df = df[int(n * (training + validation)):]
    return train_df, val_df, test_df


def data_transformation(scaler, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    # Must return a Pandas DataFrame
    train_df.loc[:, train_df.columns] = scaler.fit_transform(train_df.loc[:, train_df.columns])
    val_df.loc[:, val_df.columns] = scaler.transform(val_df.loc[:, val_df.columns])
    test_df.loc[:, test_df.columns] = scaler.transform(test_df.loc[:, test_df.columns])
    return train_df, val_df, test_df


class LstmModel:
    def __init__(self, input_width, label_width, df,
                 train_df, val_df, test_df,
                 epoch=20, units=20, layers=1,
                 dropout=0, name='LSTM',
                 ):
        # Store the raw data.
        self.df = df
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Window parameters.
        self.input_width = input_width
        self.label_width = label_width

        # Hyper parameters.
        self.epoch = epoch
        self.units = units
        self.layer = layers
        self.dropout = dropout
        self.name = name

        # Window
        self.window = WindowGenerator(
            input_width=self.input_width,
            label_width=self.label_width,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['CPU usage [MHZ]'])

        # Model
        self.model = tf.keras.models.Sequential()
        for _ in range(layers):
            if label_width == 1:
                # Shape [batch, time, features] => [batch, time, lstm_units]
                self.model.add(tf.keras.layers.LSTM(self.units, dropout=self.dropout, return_sequences=False))
            else:
                self.model.add(tf.keras.layers.LSTM(self.units, dropout=self.dropout, return_sequences=True))
        # Shape => [batch, time, features]
        self.model.add(tf.keras.layers.Dense(units=1))

    def compile_and_fit(self, patience=5):
        print('Training:')
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        # Tensorboard
        # log_dir = f'logs/fit/{self.name}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/fit/{self.name}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # self.model.build(self.window.train)
        # self.model.summary()

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsolutePercentageError(),
                                    tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(self.window.train, epochs=self.epoch,
                                 validation_data=self.window.val,
                                 callbacks=[early_stopping, tensorboard_callback])

        # Loss History figure
        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        # plt.show()
        if not os.access(os.path.join(FIGURES_PATH, self.name), os.F_OK):
            os.mkdir(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'Loss')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return history

    def prediction(self, scaler):
        print('Inference:')
        pred = self.model.predict(self.window.test_pred)
        # Convert to dataframe
        pred_df = pd.DataFrame(pred, columns=['CPU usage [MHZ]'])
        pred_df.index = self.test_df.index
        # Inverse transform
        pred_df['Day sin'] = self.test_df['Day sin']
        pred_df['Day cos'] = self.test_df['Day cos']
        pred_trf = scaler.inverse_transform(pred_df)
        pred_df_trf = pd.DataFrame(data=pred_trf, columns=self.test_df.columns, index=self.test_df.index)

        # test_trf = scaler.inverse_transform(self.test_df)
        # test_df_trf = pd.DataFrame(data=test_trf, columns=self.test_df.columns, index=self.test_df.index)
        val_mape = self.model.evaluate(self.window.val)
        # Figure
        fig = plt.figure()
        plt.grid()
        self.df['CPU usage [MHZ]'].plot(label='actual', color='k')
        pred_df_trf['CPU usage [MHZ]'].plot(label='forecast')
        plt.ylabel('CPU usage [MHz]')
        plt.title(f'Val MAPE: {val_mape}')
        plt.grid()
        plt.legend()
        if not os.access(os.path.join(FIGURES_PATH, self.name), os.F_OK):
            os.mkdir(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'forecast')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return pred_df_trf

    def evaluation(self, pred, scaler):
        performance_val = self.model.evaluate(self.window.val)
        test_trf = scaler.inverse_transform(self.test_df)
        y_true = test_trf[:, 0]
        y_pred = np.array(pred['CPU usage [MHZ]'])
        performance_test = {}
        performance_test['MAE'] = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
        performance_test['MAPE'] = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
        performance_test['MSE'] = tf.keras.metrics.mean_squared_error(y_true, y_pred)

        return performance_val, performance_test
