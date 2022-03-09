import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import datetime
import time
import math

from DataExploration.BitbrainsUtils import load_VM, plot_timeSeries, mase

FIGURES_PATH = '../Figures/Modeling/LSTM'

if not os.access(FIGURES_PATH, os.F_OK):
    os.mkdir(FIGURES_PATH)
if not os.access(FIGURES_PATH, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(FIGURES_PATH))
    exit()
else:
    print('figures saved to {}'.format(FIGURES_PATH))


class LstmModel:
    def __init__(self, input_width, label_width, df,
                 train_df, val_df, test_df,
                 epoch=100, units=20, layers=1,
                 dropout=0, batch_size=128,
                 name='LSTM', classification=False,
                 ):
        # Store the raw data.
        self.df = df
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Window parameters.
        self.input_width = input_width
        self.label_width = label_width

        # Dataframe for predictions (take test + input_length from validation set)
        self.test_pred_df = pd.concat([self.val_df.iloc[-self.input_width:, :], self.test_df])

        # Hyper parameters.
        self.epoch = epoch
        self.units = units
        self.layer = layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.name = name
        self.classification = classification

        # Model parameters
        self.train_time = 0
        self.inference_time = 0
        self.model_size = 0.
        self.mean_class = None
        self.num_classes = 0

        # Classification pre-processing
        if self.classification is True:
            labels_data, self.mean_class = reg2class(self.df['CPU usage [MHZ]'], n_classes=100)
            self.num_classes = max(labels_data) + 1
            self.train_labels, self.val_labels, self.test_labels = split_data(pd.DataFrame(labels_data))
        # Model
        self.model = tf.keras.models.Sequential()
        if self.layer == 1:
            # Shape [batch, time, features] => [batch, time, lstm_units]
            self.model.add(tf.keras.layers.LSTM(self.units, input_shape=(self.input_width, self.df.shape[1]),
                                                dropout=self.dropout, return_sequences=False))
        else:
            for _ in range(self.layer - 1):
                self.model.add(tf.keras.layers.LSTM(self.units, dropout=self.dropout, return_sequences=True))
            self.model.add(tf.keras.layers.LSTM(self.units, dropout=self.dropout, return_sequences=False))

        if self.classification is False:
            if self.label_width == 1:
                # Shape => [batch, time, features]
                self.model.add(tf.keras.layers.Dense(units=1))
            else:
                # Shape => [batch, 1, out_steps*features]
                self.model.add(
                    tf.keras.layers.Dense(units=self.label_width, kernel_initializer=tf.initializers.zeros()))
                # Shape => [batch, out_steps, features]
                self.model.add(tf.keras.layers.Reshape([self.label_width, 1]))
        else:
            self.model.add(tf.keras.layers.Dense(100, activation='relu'))
            # Units: n_classes*output_width
            self.model.add(tf.keras.layers.Dense(units=self.train[1].shape[2] * self.label_width, activation="softmax"))
            self.model.add(tf.keras.layers.Reshape([self.label_width, self.train[1].shape[2]]))

    @property
    def train(self):
        if self.classification is True:
            return self.create_dataset(self.train_df, self.train_labels)
        else:
            return self.create_dataset(self.train_df)

    @property
    def val(self):
        if self.classification is True:
            return self.create_dataset(self.val_df, self.val_labels)
        else:
            return self.create_dataset(self.val_df)

    @property
    def test(self):
        if self.classification is True:
            return self.create_dataset(self.test_df, self.test_labels)
        else:
            return self.create_dataset(self.test_df)

    @property
    def test_pred(self):
        return self.create_dataset(self.test_pred_df)

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

    def create_dataset(self, data, labels=None):
        """ Shapes the dataset ready for Keras/Tensorflow

        Parameters
        ----------
        data
        labels

        Returns
        -------

        """
        # Shape for classification
        if self.classification is True:
            labels_data = np.array(labels)
            labels_data = np.reshape(labels_data, (-1, 1))
        data = np.array(data, dtype=np.float32)
        if self.classification is False:
            labels_data = data
        # Check if the length of series is multiple of label_width
        if ((len(data) - self.input_width) % self.label_width) != 0:
            # If not: cut the series to make it multiple
            data = data[
                   :math.floor((len(data) - self.input_width) / self.label_width) * self.label_width + self.input_width,
                   :]
        # Change for multiple features
        input = []
        labels = []
        for i in range(0, len(data) - self.input_width, self.label_width):
            for j in range(data.shape[1]):
                input.append(data[i:(i + self.input_width), j])
            labels.append(labels_data[(i + self.input_width):(i + self.input_width + self.label_width), 0])
        # Shape for tensorflow: (Samples, time steps, features)
        input = np.array(input).reshape((-1, self.input_width, data.shape[1]))
        # Need to change 3rd dimension if you want to predict more than 1 output at a time
        labels = np.array(labels).reshape((-1, self.label_width, 1))
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        return input, labels

    def compile_and_fit(self, patience=100):
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        # Tensorboard
        # log_dir = f'logs/fit/{self.name}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{self.name}/tensorboard'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Save checkpoint
        checkpoint_filepath = f'logs/{self.name}/checkpoints'
        if not os.access(checkpoint_filepath, os.F_OK):
            os.makedirs(checkpoint_filepath)
        if not os.access(checkpoint_filepath, os.W_OK):
            print('Cannot write to {}, please fix it.'.format(checkpoint_filepath))
            exit()
        checkpoint_filepath = os.path.join(checkpoint_filepath, 'best_model.hdf5')

        if self.classification is False:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_mean_absolute_error',
                mode='min',
                save_best_only=True)
        else:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_categorical_accuracy',
                mode='min',
                save_best_only=True)

        print(f'Input shape (batch, time, features): {self.train[0].shape}')
        print(f'Labels shape (batch, time, features): {self.train[1].shape}')
        print(f'Output shape:{self.model(self.train[0]).shape}')
        self.model.build(self.example[0].shape)
        self.model.summary()

        if self.classification is False:
            self.model.compile(loss=tf.losses.MeanSquaredError(),
                               optimizer=tf.optimizers.Adam(),
                               metrics=tf.metrics.MeanAbsoluteError()
                               )
        else:
            self.model.compile(loss=tf.losses.CategoricalCrossentropy(),
                               optimizer=tf.optimizers.Adam(),
                               metrics=tf.metrics.CategoricalAccuracy()
                               )

        # Trace time for training
        print('Beginning training')
        t_start = time.perf_counter()
        history = self.model.fit(self.train[0],
                                 self.train[1],
                                 epochs=self.epoch,
                                 batch_size=self.batch_size,
                                 validation_data=(self.val[0], self.val[1]),
                                 callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback])

        self.train_time = time.perf_counter() - t_start
        self.model_size = os.stat(checkpoint_filepath).st_size
        print("Training has completed:", f'{self.train_time:.2f} sec')

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
        t_start = time.perf_counter()
        pred = self.model.predict(self.test_pred[0], batch_size=self.batch_size)
        self.inference_time = time.perf_counter() - t_start
        print("Inference time:", f'{self.inference_time:.2f} sec')
        # Reshape
        pred = np.argmax(pred, axis=2)
        pred = np.reshape(pred, (-1, 1))
        if self.classification is True:
            pred = class2num(pred, self.mean_class)
        # Cut test_df
        self.test_df = self.test_df.iloc[:pred.shape[0], :]

        # Convert to dataframe
        pred_df = pd.DataFrame(pred, columns=['CPU usage [MHZ]'])
        pred_df.index = self.test_df.index
        # Inverse transform
        pred_trf = scaler.inverse_transform(pred_df)
        pred_df_trf = pd.DataFrame(data=pred_trf, columns=['CPU usage [MHZ]'], index=self.test_df.index)
        if self.classification is True:
            pred_df_trf = pred_df
        # Whole set
        # Convert to dataframe
        df_trf = scaler.inverse_transform(self.df)
        df_df_trf = pd.DataFrame(data=df_trf, columns=self.df.columns, index=self.df.index)
        # Test set
        test_trf = scaler.inverse_transform(self.test_df)
        test_df_trf = pd.DataFrame(data=test_trf, columns=self.test_df.columns, index=self.test_df.index)
        val_mae = self.model.evaluate(self.val[0], self.val[1])

        # Figure forecast
        # Define default kwargs
        defaultKwargs = {'marker': 'o',
                         'linestyle': '',
                         'alpha': 0.3,
                         'markersize': 2}
        kwargs_forecast = {'marker': 'o',
                           'linestyle': '',
                           'alpha': 0.5,
                           'markersize': 2,
                           'color': 'tab:orange'}
        fig = plt.figure(dpi=200)
        plt.grid()
        self.df['CPU usage [MHZ]'].plot(label='actual', color='k', **defaultKwargs)
        pred_df_trf['CPU usage [MHZ]'].plot(label='forecast', **kwargs_forecast)
        plt.ylabel('CPU usage [MHz]')
        plt.title(f'Val MAE: {val_mae[1]:.3f}')
        plt.grid()
        plt.legend()
        if not os.access(os.path.join(FIGURES_PATH, self.name), os.F_OK):
            os.mkdir(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'forecast')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure zoom
        fig = plt.figure(dpi=200)
        plt.grid()
        test_df_trf['CPU usage [MHZ]'].plot(label='actual', color='k', **defaultKwargs)
        pred_df_trf['CPU usage [MHZ]'].plot(label='forecast', **kwargs_forecast)
        plt.ylabel('CPU usage [MHz]')
        plt.title(f'Val MAE:{val_mae[1]:.3f}')
        plt.grid()
        plt.legend()
        if not os.access(os.path.join(FIGURES_PATH, self.name), os.F_OK):
            os.mkdir(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'forecast_zoom')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return pred_df_trf

    def evaluation(self, pred, scaler):
        test_trf = scaler.inverse_transform(self.test_df)
        train_trf = scaler.inverse_transform(self.train_df)
        y_true = np.array(test_trf[:, 0])
        y_pred = np.array(pred['CPU usage [MHZ]'])
        y_train = np.array(train_trf[:, 0])
        metrics_dic = {'MAE': np.array(tf.keras.metrics.mean_absolute_error(y_true, y_pred)),
                       'MAPE': np.array(tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)),
                       'RMSE': np.sqrt(np.array(tf.keras.metrics.mean_squared_error(y_true, y_pred))),
                       'MASE': mase(y_true, y_pred, y_train),
                       'train_time [s]': self.train_time,
                       'inference_time [s]': self.inference_time,
                       'model_size [B]': self.model_size
                       }

        # Save metrics
        metrics = pd.DataFrame.from_dict(metrics_dic, orient='index')
        try:
            filename = os.path.join('logs', self.name, 'metrics.txt')
            metrics.to_csv(filename)
            # file = open(filename, 'wt')
            # file.write(str(metrics))
            # file.close()
        except:
            print("Unable to write to file")
        return metrics


def add_daily_info(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_s = df.index.map(pd.Timestamp.timestamp)

    day = 60 * 24 / 5
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Hour'] = np.array(df.index.floor(freq='H').hour)
    return df


def split_data(df: pd.DataFrame, training: float = 0.7, validation: float = 0.2, test: float = 0.1):
    """ Split the dataset in train, validation and test sets

    Parameters
    ----------
    df
    training
    validation
    test

    Returns
    -------

    """
    n = len(df)
    df_copy = df.copy()
    train_df = df_copy[0:int(n * training)]
    val_df = df_copy[int(n * training):int(n * (training + validation))]
    test_df = df_copy[int(n * (training + validation)):]
    return train_df, val_df, test_df


def data_transformation(scaler, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """ Performs data pre-processing according to scaler

    Parameters
    ----------
    scaler
    train_df
    val_df
    test_df

    Returns
    -------

    """
    # Must return a Pandas DataFrame
    train_df.loc[:, train_df.columns] = scaler.fit_transform(train_df.loc[:, train_df.columns])
    val_df.loc[:, val_df.columns] = scaler.transform(val_df.loc[:, val_df.columns])
    test_df.loc[:, test_df.columns] = scaler.transform(test_df.loc[:, test_df.columns])
    return train_df, val_df, test_df


def reg2class(df, n_classes):
    """ Regression to Classification problem
        Return the labels (0 to num_classes) & the mean per class
    Parameters
    ----------
    df
    n_classes

    Returns
    -------

    """
    label_encoder = LabelEncoder()
    # Create n_classes in the df
    classes = pd.cut(df, n_classes, retbins=True)
    # Class labels
    labels = label_encoder.fit_transform(classes[0])
    # Return also numeric value per class
    boundaries = classes[1]
    mean_class = []
    for i in range(len(boundaries) - 1):
        mean_class.append(0.5 * (boundaries[i] + boundaries[i + 1]))
    mean_class = np.array(mean_class)
    # Return only classes present in the dataset
    mean_class = mean_class[np.array(pd.cut(df, bins=n_classes).value_counts(sort=False)) != 0]
    return labels, mean_class


def class2num(y, mean_class):
    """ Convert classes to numeric output

    Parameters
    ----------
    y
    mean_class

    Returns
    -------

    """
    return mean_class[y]

# Do not need it anymore
# class WindowGenerator:
#     def __init__(self, input_width, label_width, shift,
#                  train_df, val_df, test_df, batch_size,
#                  label_columns=None, classification=False):
#         # Store the raw data.
#         self.train_df = train_df
#         self.val_df = val_df
#         self.test_df = test_df
#
#         # Work out the label column indices.
#         self.label_columns = label_columns
#         if label_columns is not None:
#             self.label_columns_indices = {name: i for i, name in
#                                           enumerate(label_columns)}
#         self.column_indices = {name: i for i, name in
#                                enumerate(train_df.columns)}
#
#         # Work out the window parameters.
#         self.input_width = input_width
#         self.label_width = label_width
#         self.shift = shift
#         self.bath_size = batch_size
#
#         self.total_window_size = input_width + shift
#
#         self.input_slice = slice(0, input_width)
#         self.input_indices = np.arange(self.total_window_size)[self.input_slice]
#
#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
#
#         # Dataframe for predictions (take test + input_length from validation set)
#         self.test_pred_df = pd.concat([self.val_df.iloc[-self.input_width:, :], self.test_df])
#
#     def __repr__(self):
#         return '\n'.join([
#             f'Total window size: {self.total_window_size}',
#             f'Input indices: {self.input_indices}',
#             f'Label indices: {self.label_indices}',
#             f'Label column name(s): {self.label_columns}'])
#
#     def split_window(self, features):
#         inputs = features[:, self.input_slice, :]
#         labels = features[:, self.labels_slice, :]
#         if self.label_columns is not None:
#             labels = tf.stack(
#                 [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#                 axis=-1)
#
#         # Slicing doesn't preserve static shape information, so set the shapes
#         # manually. This way the `tf.data.Datasets` are easier to inspect.
#         inputs.set_shape([None, self.input_width, None])
#         labels.set_shape([None, self.label_width, None])
#
#         return inputs, labels
#
#     def make_dataset(self, data):
#         data = np.array(data, dtype=np.float32)
#         ds = tf.keras.utils.timeseries_dataset_from_array(
#             data=data,
#             targets=None,
#             sequence_length=self.total_window_size,
#             sequence_stride=1,
#             shuffle=True,
#             batch_size=self.bath_size,
#         )
#
#         ds = ds.map(self.split_window)
#         return ds
#
#     def create_dataset(self, data):
#         data = np.array(data, dtype=np.float32)
#         # Check if the length of series is multiple of label_width
#         if ((len(data) - self.input_width) % self.label_width) != 0:
#             # If not: cut the series to make it multiple
#             data = data[
#                    :math.floor((len(data) - self.input_width) / self.label_width) * self.label_width + self.input_width,
#                    :]
#         # Change for multiple features
#         input = []
#         labels = []
#         for i in range(0, len(data) - self.input_width, self.label_width):
#             for j in range(data.shape[1]):
#                 input.append(data[i:(i + self.input_width), j])
#             labels.append(data[(i + self.input_width):(i + self.input_width + self.label_width), 0])
#         # Shape for tensorflow: (Samples, time steps, features)
#         input = np.array(input).reshape((-1, self.input_width, data.shape[1]))
#         # Need to change 3rd dimension if you want to predict more than 1 output at a time
#         labels = np.array(labels).reshape((-1, self.label_width, 1))
#         return input, labels
#
#     @property
#     def train(self):
#         return self.create_dataset(self.train_df)
#
#     @property
#     def val(self):
#         return self.create_dataset(self.val_df)
#
#     @property
#     def test(self):
#         return self.create_dataset(self.test_df)
#
#     @property
#     def test_pred(self):
#         return self.create_dataset(self.test_pred_df)
#
#     @property
#     def example(self):
#         """Get and cache an example batch of `inputs, labels` for plotting."""
#         result = getattr(self, '_example', None)
#         if result is None:
#             # No example batch was found, so get one from the `.train` dataset
#             result = next(iter(self.train))
#             # And cache it for next time
#             self._example = result
#         return result
