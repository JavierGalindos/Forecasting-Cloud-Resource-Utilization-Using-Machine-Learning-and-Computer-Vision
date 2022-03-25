import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import datetime as dt
import time
import math
import io
import cv2
import random

from DataExploration.BitbrainsUtils import *

FIGURES_PATH = '../Figures/Modeling/ConvLSTM'

if not os.access(FIGURES_PATH, os.F_OK):
    os.mkdir(FIGURES_PATH)
if not os.access(FIGURES_PATH, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(FIGURES_PATH))
    exit()
else:
    print('figures saved to {}'.format(FIGURES_PATH))

DATASET_PATH = '../Datasets/fastStorage/images'

if not os.access(DATASET_PATH, os.F_OK):
    os.mkdir(DATASET_PATH)
if not os.access(DATASET_PATH, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(DATASET_PATH))
    exit()
else:
    print('figures saved to {}'.format(DATASET_PATH))

# Set the seed for reproducibility
seed_constant = 42
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Global figure size and resolution. Gives image of size 64x64
dpi = 8
figsize = (8, 8)


class ConvLSTMModel:
    def __init__(self, input_width, label_width, df,
                 train_df, val_df, test_df,
                 epoch=100, batch_size=64,
                 n_frames=1,
                 name='ConvLSTM',
                 model_path=None,
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
        self.batch_size = batch_size
        self.name = name
        self.n_frames = n_frames

        # Model parameters
        self.train_time = 0
        self.inference_time = 0
        self.model_size = 0.

        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self.get_model()

    @property
    def train(self):
        return self.create_dataset(self.train_df)

    @property
    def val(self):
        return self.create_dataset(self.val_df)

    @property
    def test(self):
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

    @staticmethod
    def create_image_numpy(data, width, height=100):
        data = np.squeeze(data)
        # Get the height of each column (0-100)
        level = np.round(data * 100)
        # Invert y-axis from bottom to top
        level = 100 - level - 1
        # Create the image
        img = np.zeros((height, width))
        # Fill with ones the corresponding level
        for i in range(width):
            img[int(level[i]), i] = 255
        return img

    @staticmethod
    def create_image_matplotlib(data):
        global figsize, dpi
        # Input
        fig = full_frame(figsize, dpi)
        plt.ylim(0, 1)  # set y-axis limits
        plt.plot(data, ',', color='black')
        # Save images temporally in the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        # Get image from buffer
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)
        # Read with OpenCV to get numpy array
        img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        return img

    def create_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        # Check if the length of series is multiple of label_width
        if ((len(data) - (self.input_width + self.label_width * (self.n_frames - 1))) % self.label_width) != 0:
            # If not: trim the series to make it multiple
            data = data[
                   :math.floor(
                       (len(data) - (self.input_width + self.label_width * (self.n_frames - 1))) / self.label_width)
                    * self.label_width + self.input_width + self.label_width * (self.n_frames - 1),
                   :]

        # Generate the images
        # Overlapping c=(input_width-label_width)/input_width
        # Input images (shift label with between samples)
        input = []
        labels = []
        frames = []
        for i in range(0, len(data) - (self.input_width + self.label_width * (self.n_frames - 1)), self.label_width):
            # Input
            # Create n_frames images for the input. Always same overlapping
            for j in range(0, self.n_frames):
                img = self.create_image_numpy(
                    data[(i + j * self.label_width):(i + j * self.label_width + self.input_width), :], self.input_width,
                    100)
                # Normalize image
                img_normalized = img / 255.
                frames.append(img_normalized)
            # Append frames and clean the variable
            input.append(frames)
            frames = []

            # Labels
            # j = n_frames + 1 (to save previous position and get next image
            img = self.create_image_numpy(
                data[(i + j * self.label_width + self.label_width):(
                        i + j * self.label_width + self.input_width + self.label_width), :],
                self.input_width, 100)
            # Normalize image
            img_normalized = img / 255.
            labels.append(img_normalized)

        input = np.array(input)
        input = np.expand_dims(input, axis=4)
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=(1, 4))

        return input, labels

    def create_image_dataset(self, data, image_path):
        # TODO: change for numpy images
        data = np.array(data, dtype=np.float32)
        # Check if the length of series is multiple of label_width
        if ((len(data) - self.input_width) % self.label_width) != 0:
            # If not: trim the series to make it multiple
            data = data[
                   :math.floor((len(data) - self.input_width) / self.label_width) * self.label_width + self.input_width,
                   :]

        # Generate the images
        # Overlapping c=(input_width-label_width)/input_width
        j = 0
        for i in range(0, len(data) - self.input_width, self.label_width):
            # Input images (shift label with between samples)
            fig = full_frame(figsize, dpi)
            plt.ylim(0, 1)  # set y-axis limits
            plt.plot(data[i:(i + self.input_width), :], ',', color='black')
            image_path_save = os.path.join(image_path, 'input')
            # Create the folder whether not exists
            if not os.path.exists(image_path_save):
                os.makedirs(image_path_save)
            plt.savefig(os.path.join(image_path_save, f'{j}.png'), format="png")
            plt.close(fig)

            # Label images (shift label with between samples)
            fig = full_frame(figsize, dpi)
            plt.ylim(0, 1)  # set y-axis limits
            plt.plot(data[(i + self.label_width):(i + self.input_width + self.label_width), :], ',', color='black')
            image_path_save = os.path.join(image_path, 'labels')
            # Create the folder whether not exists
            if not os.path.exists(image_path_save):
                os.makedirs(image_path_save)
            plt.savefig(os.path.join(image_path_save, f'{j}.png'), format="png")
            plt.close(fig)
            j += 1

    def get_model(self):
        tf.config.set_soft_device_placement(True)
        # Construct the input layer with no definite frame size.
        # inp = tf.keras.layers.Input(shape=(None, *self.train[0].shape[2:]))
        inp = tf.keras.layers.Input(shape=(None, *self.train[0].shape[2:]))

        # We will construct 3 `ConvLSTM2D` layers with batch normalization,
        # followed by a `Conv3D` layer for the spatiotemporal outputs.
        x = tf.keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=True,
            activation="relu",
            data_format="channels_last",
        )(inp)
        # x = tf.keras.layers.Reshape([100, self.input_width, 64])(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Reshape([1, 100, self.input_width, 64])(x)
        x = tf.keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            data_format="channels_last",
        )(x)
        # x = tf.keras.layers.Reshape([100, self.input_width, 64])(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Reshape([1, 100, self.input_width, 64])(x)
        x = tf.keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
            data_format="channels_last",
        )(x)
        x = tf.keras.layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same",
        )(x)

        # Next, we will build the complete model and compile it.
        model = tf.keras.models.Model(inp, x)
        model.compile(
            loss=tf.keras.losses.BinaryFocalCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
        )
        return model

    def compile_and_fit(self):
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=10,
                                                          mode='min',
                                                          restore_best_weights=True)

        # Tensorboard
        # log_dir = f'logs/fit/{self.name}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/ConvLSTM/{self.name}/tensorboard'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

        print(f'Input shape (batch_size, num_frames, width, height, channels): {self.train[0].shape}')
        print(f'Labels shape (batch_size, num_frames, width, height, channels): {self.train[1].shape}')
        # print(f'Output shape:{self.model(self.train[0]).shape}')
        self.model.build(self.example[0].shape)
        self.model.summary()

        # Trace time for training
        print('Beginning training')
        t_start = time.perf_counter()
        history = self.model.fit(self.train[0],
                                 self.train[1],
                                 epochs=self.epoch,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 validation_data=(self.val[0], self.val[1]),
                                 callbacks=[early_stopping, tensorboard_callback, reduce_lr])

        self.train_time = time.perf_counter() - t_start
        print("Training has completed:", f'{self.train_time:.2f} sec')

        # Save the best model (getting the best with early_stopping callback)
        # Check the path exists
        model_filepath = f'logs/ConvLSTM/{self.name}/checkpoints'
        if not os.access(model_filepath, os.F_OK):
            os.makedirs(model_filepath)
        if not os.access(model_filepath, os.W_OK):
            print('Cannot write to {}, please fix it.'.format(model_filepath))
            exit()

        # Get the loss and accuracy from model_evaluation_history.
        # model_evaluation_loss, model_evaluation_accuracy = history
        # Define the string date format.
        # Get the current Date and Time in a DateTime Object.
        # Convert the DateTime object to string according to the style mentioned in date_time_format string.
        date_time_format = '%Y_%m_%d__%H_%M_%S'
        current_date_time_dt = dt.datetime.now()
        current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
        # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
        # model_file_name = f'ConvLSTM_model_{current_date_time_string}_Loss_{model_evaluation_loss}_Accuracy_{model_evaluation_accuracy}.hdf5'
        model_file_name = f'ConvLSTM_model_{current_date_time_string}.hdf5'
        model_filepath = os.path.join(model_filepath, model_file_name)
        # Save your Model.
        self.model.save(model_filepath)
        self.model_size = os.stat(model_filepath).st_size

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
            os.makedirs(os.path.join(FIGURES_PATH, self.name))
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

        # Ground truth
        gt = self.test[1]

        # Test_pred dataset
        test_input = self.test_pred[0]
        test_label = self.test_pred[1]
        # Construct a figure for the original and new frames.
        fig, axes = plt.subplots(4, 5, figsize=(20, 20))
        # Plot the original frames.
        axes[0][2].set_title('Input')
        for idx, ax in enumerate(axes[0]):
            ax.imshow(np.squeeze(test_input[idx]), cmap="gray")
            # ax.set_title(f"Sample {idx}")
            ax.axis("off")
        # Plot the new frames.
        axes[1][2].set_title('Labels')
        for idx, ax in enumerate(axes[1]):
            ax.imshow(np.squeeze(test_label[idx]), cmap="gray")
            # ax.set_title(f"Sample {idx}")
            ax.axis("off")
        axes[2][2].set_title('Output')
        # Prediction
        for idx, ax in enumerate(axes[2]):
            ax.imshow(np.squeeze(pred[idx]), cmap="gray")
            # ax.set_title(f"Sample {idx}")
            ax.axis("off")
        # Prediction binarize
        axes[3][2].set_title('Output binarized')
        for idx, ax in enumerate(axes[3]):
            ax.imshow(np.squeeze(self.binarize_image(pred[idx])), cmap="gray")
            # ax.set_title(f"Sample {idx}")
            ax.axis("off")
        # Save the figure
        if not os.access(os.path.join(FIGURES_PATH, self.name), os.F_OK):
            os.makedirs(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'raw_output')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure for forecasting part of image
        img_pred = self.extract_forecast_from_image(pred)
        img_gt = self.extract_forecast_from_image(gt)
        # Binarize the image
        img_pred_bin = self.binarize_image(img_pred)
        fig, axes = plt.subplots(2, 1, figsize=(15, 6))
        plt.suptitle('Test set: ground truth vs prediction', fontsize=16)
        # Ground Truth
        axes[0].imshow(img_gt, cmap="gray")
        axes[0].set_title('Ground truth')
        axes[0].axis("off")
        # Prediction
        axes[1].imshow(img_pred_bin, cmap="gray")
        axes[1].set_title('Prediction')
        axes[1].axis("off")
        save_path = os.path.join(FIGURES_PATH, self.name, 'gt_vs_pred')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure with numeric data
        pred_df_trf = self.image2series(img_pred, scaler)
        # Test set
        test_trf = scaler.inverse_transform(self.test_df)
        test_df_trf = pd.DataFrame(data=test_trf, columns=self.test_df.columns, index=self.test_df.index)

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
        plt.title(f'Actual vs Forecast')
        plt.grid()
        plt.legend()
        save_path = os.path.join(FIGURES_PATH, self.name, 'forecast')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure zoom
        fig = plt.figure(dpi=200)
        plt.grid()
        test_df_trf['CPU usage [MHZ]'].plot(label='actual', color='k', **defaultKwargs)
        pred_df_trf['CPU usage [MHZ]'].plot(label='forecast', **kwargs_forecast)
        plt.ylabel('CPU usage [MHz]')
        plt.title(f'Actual vs Forecast (Zoom)')
        plt.grid()
        plt.legend()
        save_path = os.path.join(FIGURES_PATH, self.name, 'forecast_zoom')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        return pred, img_pred_bin, pred_df_trf

    def evaluation(self, pred, scaler):
        test_trf = scaler.inverse_transform(self.test_df)
        train_trf = scaler.inverse_transform(self.train_df)
        y_true = np.array(test_trf[:len(pred), 0])
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
        print(metrics)
        try:
            filename = os.path.join('./logs/ConvLSTM', self.name, 'metrics.txt')
            metrics.to_csv(filename)
        except:
            print("Unable to write to file")

        # Errors boxplot
        errors = self.errors_boxplot(pred, scaler)
        return metrics

    def errors_boxplot(self, pred, scaler):
        test_trf = scaler.inverse_transform(self.test_df)
        y_true = np.array(test_trf[:len(pred), 0])
        y_pred = np.array(pred['CPU usage [MHZ]'])
        # Create a dataframe of errors
        errors_dic = {'MAE': mae_array(y_true, y_pred),
                      'MAPE': mape_array(y_true, y_pred),
                      'RMSE': rmse_array(y_true, y_pred),
                      }
        errors = pd.DataFrame.from_dict(errors_dic, orient='index')
        errors = errors.T
        print(errors.describe())
        ax = sns.boxplot(data=errors).set(title='Errors box-plot',
                                          ylabel='Error')
        if not os.access(os.path.join(FIGURES_PATH, self.name), os.F_OK):
            os.mkdir(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'errors_boxplot')
        plt.savefig(save_path, bbox_inches='tight')
        # Save the errors
        try:
            filename = os.path.join('./logs/ConvLSTM', self.name, 'errors.csv')
            errors.to_csv(filename)
        except:
            print("Unable to write to file")
        return errors

    @staticmethod
    def binarize_image(img):
        # Creates a binary image
        # Squeeze if needed
        img = np.squeeze(img)
        # Take the brightest pixel of each column
        idx = np.argmax(img, axis=0)
        # New black image
        new_img = np.zeros_like(img)
        # Draw a white pixel in the brightest value of each column
        for i in range(new_img.shape[1]):
            new_img[idx[i], i] = 255
        return new_img

    def extract_forecast_from_image(self, pred):
        # Get the forecasting part of each image (last label_width time steps) and concatenate
        pred = np.squeeze(pred)
        # Shape (frame, height, width)
        img_pred = []
        for i in range(pred.shape[0]):
            img_pred.append(pred[i, :, -self.label_width:])
        img_pred = np.concatenate(img_pred, axis=1)
        return img_pred

    def image2series(self, img, scaler):
        # Take the brightest pixel of each column
        idx = np.argmax(img, axis=0)
        # Numeric (undo preprocessing)
        pred_numeric = 100 - 1 - idx
        pred_numeric = pred_numeric / 100

        # Check if the prediction is longer than the test set (then trim it)
        if len(pred_numeric) > len(self.test_df):
            pred_numeric = pred_numeric[:len(self.test_df)]

        # Convert to dataframe
        pred_df = pd.DataFrame(pred_numeric, columns=['CPU usage [MHZ]'])
        pred_df.index = self.test_df.index[:len(pred_df)]
        # Inverse transform
        pred_trf = scaler.inverse_transform(pred_df)
        pred_df_trf = pd.DataFrame(data=pred_trf, columns=['CPU usage [MHZ]'], index=pred_df.index)
        # pred_df_trf = pred_df_trf.shift(periods=-self.label_width)
        return pred_df_trf


def full_frame(figsize=(8.0, 8.0), dpi=8):
    """
    Helper function to create figure with no axes,borders,frame etc.
    Parameters
    ----------
    figsize:
        Width, height in inches
    dpi:
        The resolution of the figure in dots-per-inch.

    Returns
    -------
    `~matplotlib.figure.Figure`
        figure instance
    """
    mpl.rcParams['savefig.pad_inches'] = 0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    return fig
