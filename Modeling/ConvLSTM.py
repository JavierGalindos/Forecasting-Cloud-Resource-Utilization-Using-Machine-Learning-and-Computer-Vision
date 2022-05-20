import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import datetime as dt
import time
import math
import io
import cv2
import random
from skimage.exposure import rescale_intensity

from DataExploration.BitbrainsUtils import *
from model_zoo import get_model

FIGURES_PATH = '../Figures/Modeling/ConvLSTM'
plt.rcParams['font.size'] = '13'

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
                 model_name,
                 image_width=None,
                 epoch=100, batch_size=16,
                 n_frames=1,
                 name='ConvLSTM',
                 model_path=None,
                 numeric=False,
                 overlapping=None,
                 ):
        # Store the raw data.
        self.df = df
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.overlapping = overlapping
        # Adapt to image_width different to input_width
        if image_width is None:
            self.image_width = self.input_width
        else:
            self.image_width = image_width
        # Ratio between image_with and input_width
        self.ratio = int(self.input_width / self.image_width)

        # Hyper parameters.
        self.model_name = model_name
        self.model_path = model_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.name = name
        self.n_frames = n_frames
        self.numeric = numeric

        # Dataframe for predictions (take test + input_length from validation set)
        self.test_pred_df = pd.concat(
            [self.val_df.iloc[-(self.input_width + self.label_width * (self.n_frames - 1)):, :], self.test_df])

        # Model parameters
        self.train_time = 0
        self.inference_time = 0
        self.model_size = 0.

        # Add overlapping as parameters
        # Forecasting horizon = (1 - overlapping)* input = input - overlapping*input
        if self.overlapping is not None:
            self.label_width = self.input_width - int(overlapping * self.input_width)

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

    def create_image_numpy(self, data, width, height=100):
        if self.ratio != 1:
            assert "Do not use create_image_numpy if ratio > 1"
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
    def create_image_matplotlib(data, width, height=100):
        # Size = figsize*dpi = (100,64)
        # Input
        fig = full_frame(figsize=(width / 100, height / 100), dpi=100)
        # plt.style.use('dark_background')
        plt.ylim(0, 1)  # set y-axis limits
        plt.plot(data, ',', color='black', linewidth=0.1)
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
        # img = rescale_int(img)
        img = 255 - img
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

        # Generate the images (change between create_image_numpy and create_image_matplotlib)
        # Overlapping c=(input_width-label_width)/input_width
        # Input images (shift label with between samples)
        input = []
        labels = []
        frames = []
        for i in range(0, len(data) - (self.input_width + self.label_width * (self.n_frames - 1)), self.label_width):
            # Input
            # Create n_frames images for the input. Always same overlapping
            for j in range(0, self.n_frames):
                img = self.create_image_matplotlib(
                    data[(i + j * self.label_width):(i + j * self.label_width + self.input_width), :], self.image_width,
                    100)
                # Normalize image
                img_normalized = img / 255.
                frames.append(img_normalized)
            # Append frames and clean the variable
            input.append(frames)
            frames = []

            # Labels
            if self.numeric is False:
                # j = n_frames + 1 (to save previous position and get next image
                if self.model_name == "video":
                    img = self.create_image_matplotlib(
                        data[(i + j * self.label_width + self.label_width):(
                                i + j * self.label_width + self.input_width + self.label_width), :],
                        self.image_width, 100)
                    # Normalize image
                    img_normalized = img / 255.
                    frames.append(img_normalized)
                    labels.append(frames)
                    frames = []
                else:
                    for j in range(0, self.n_frames):
                        img = self.create_image_matplotlib(
                            data[(i + j * self.label_width + self.label_width):(
                                    i + j * self.label_width + self.input_width + self.label_width), :],
                            self.image_width, 100)
                        # Normalize image
                        img_normalized = img / 255.
                        frames.append(img_normalized)
                    labels.append(frames)
                    frames = []
            else:
                labels.append(data[(i + j * self.label_width + self.input_width):(
                        i + self.input_width + j * self.label_width + self.label_width), 0])

        # input = np.array(input).astype(int)  # Change to int
        input = np.array(input)
        input = np.expand_dims(input, axis=4)
        if self.numeric is False:
            # labels = np.array(labels).astype(int)
            labels = np.array(labels)
            labels = np.expand_dims(labels, axis=4)
        else:
            labels = np.array(labels).reshape((-1, self.label_width, 1))

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

    def compile_and_fit(self):
        # Load model
        if self.model_path is not None:
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.model = get_model(self, self.model_name)
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=20,
                                                          mode='min',
                                                          restore_best_weights=True)

        # Tensorboard
        # log_dir = f'logs/fit/{self.name}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/ConvLSTM/{self.model_name}/{self.name}/tensorboard'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=15)

        print("Shapes:")
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
                                 # shuffle=False,
                                 validation_data=(self.val[0], self.val[1]),
                                 callbacks=[early_stopping, tensorboard_callback, reduce_lr])

        self.train_time = time.perf_counter() - t_start
        print("Training has completed:", f'{self.train_time:.2f} sec')

        # Save the best model (getting the best with early_stopping callback)
        # Check the path exists
        model_filepath = f'logs/ConvLSTM/{self.model_name}/{self.name}/checkpoints'
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
        if not os.access(os.path.join(FIGURES_PATH, self.model_name, self.name), os.F_OK):
            os.makedirs(os.path.join(FIGURES_PATH, self.model_name, self.name))
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'Loss')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        # Save model structure
        tf.keras.utils.plot_model(self.model,
                                  to_file=os.path.join(FIGURES_PATH, self.model_name, self.name, 'model_structure.png'),
                                  show_shapes=True, show_layer_names=True)
        return history

    def prediction(self, scaler):
        print('Inference:')
        t_start = time.perf_counter()
        pred = self.model.predict(self.test_pred[0], batch_size=self.batch_size)
        self.inference_time = time.perf_counter() - t_start
        print("Inference time:", f'{self.inference_time:.2f} sec')

        if self.numeric is False:
            # [-1] to take the last frame when having multiple
            pred = pred[:, -1, ...]
            # Ground truth
            gt = self.test[1][:, -1, ...]
            # Test_pred dataset
            test_input = self.test_pred[0][:, -1, ...]
            test_label = self.test_pred[1][:, -1, ...]

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
            if not os.access(os.path.join(FIGURES_PATH, self.model_name, self.name), os.F_OK):
                os.makedirs(os.path.join(FIGURES_PATH, self.model_name, self.name))
            save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'raw_output')
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
            save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'gt_vs_pred')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            # Figure with numeric data
            pred_df_trf = self.image2series(img_pred, scaler)
            # Test set
            test_trf = scaler.inverse_transform(self.test_df)
            test_df_trf = pd.DataFrame(data=test_trf, columns=self.test_df.columns, index=self.test_df.index)
        else:
            # Reshape
            pred = np.reshape(pred, (-1, 1))
            # Convert to dataframe
            pred_df = pd.DataFrame(pred, columns=['CPU usage [MHZ]'])
            pred_df.index = self.test_df.index[:len(pred_df)]
            # Inverse transform
            pred_trf = scaler.inverse_transform(pred_df)
            pred_df_trf = pd.DataFrame(data=pred_trf, columns=['CPU usage [MHZ]'],
                                       index=self.test_df.index[:len(pred_df)])
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
                         'linestyle': '-',
                         'alpha': 0.3,
                         'markersize': 2,
                         'linewidth': 2.0,
                         }
        kwargs_forecast = {'marker': 'o',
                           'linestyle': '-',
                           'alpha': 0.5,
                           'markersize': 2,
                           'color': 'tab:green',
                           'linewidth': 2.0,}
        fig = plt.figure(dpi=200, figsize=(20, 3))
        plt.grid()
        self.df['CPU usage [MHZ]'].plot(label='actual', color='k', **defaultKwargs)
        pred_df_trf['CPU usage [MHZ]'].plot(label='forecast', **kwargs_forecast)
        plt.ylabel('CPU usage [MHz]')
        plt.title(f'Actual vs Forecast')
        plt.grid()
        plt.legend()
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'forecast')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure zoom
        fig = plt.figure(dpi=200, figsize=(15, 2))
        # plt.grid()
        test_df_trf['CPU usage [MHZ]'].plot(label='actual', color='tab:grey', **defaultKwargs)
        pred_df_trf['CPU usage [MHZ]'].plot(label='forecast', **kwargs_forecast)
        plt.ylabel('CPU usage [MHz]')
        # plt.title(f'Actual vs Forecast (Zoom)')
        # plt.grid()
        plt.legend()
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'forecast_zoom')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure GT vs prediction (test)
        # Dots
        defaultKwargs = {'marker': 'o',
                         'linestyle': '',
                         'alpha': 0.6,
                         'markersize': 2}
        kwargs_forecast = {'marker': 'o',
                           'linestyle': '',
                           'alpha': 0.6,
                           'markersize': 2,
                           'color': 'tab:orange'}
        # Construct a figure for the original and new frames.
        fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True, sharey=True)
        plt.suptitle('Test set: GT vs prediction', fontsize=16)
        # Ground Truth
        axes[0].plot(test_df_trf['CPU usage [MHZ]'], label='actual', color='k', **defaultKwargs)
        axes[0].set_title('Ground truth')
        axes[0].set_ylabel('CPU usage [MHz]')
        # Prediction
        axes[1].plot(pred_df_trf['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
        axes[1].set_title('Prediction')
        axes[1].set_ylabel('CPU usage [MHz]')
        axes[1].set_xlabel('Time')
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'gt_vs_pred_dots')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Lines and dots
        defaultKwargs = {'marker': 'o',
                         'linestyle': '-',
                         'alpha': 0.6,
                         'markersize': 2}
        kwargs_forecast = {'marker': 'o',
                           'linestyle': '-',
                           'alpha': 0.6,
                           'markersize': 2,
                           'color': 'tab:orange'}
        # Construct a figure for the original and new frames.
        fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True, sharey=True)
        plt.suptitle('Test set: GT vs prediction', fontsize=16)
        # Ground Truth
        axes[0].plot(test_df_trf['CPU usage [MHZ]'], label='actual', color='k', **defaultKwargs)
        axes[0].set_title('Ground truth')
        axes[0].set_ylabel('CPU usage [MHz]')
        # Prediction
        axes[1].plot(pred_df_trf['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
        axes[1].set_title('Prediction')
        axes[1].set_ylabel('CPU usage [MHz]')
        axes[1].set_xlabel('Time')
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'gt_vs_pred_lines')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure GT vs prediction (full)
        # Dots
        defaultKwargs = {'marker': 'o',
                         'linestyle': '',
                         'alpha': 0.6,
                         'markersize': 2}
        kwargs_forecast = {'marker': 'o',
                           'linestyle': '',
                           'alpha': 0.6,
                           'markersize': 2,
                           'color': 'tab:orange'}
        # Construct a figure for the original and new frames.
        fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True, sharey=True)
        plt.suptitle('Test set: GT vs prediction', fontsize=16)
        # Ground Truth
        axes[0].plot(self.df['CPU usage [MHZ]'], label='actual', color='k', **defaultKwargs)
        axes[0].set_title('Ground truth')
        axes[0].set_ylabel('CPU usage [MHz]')
        # Prediction
        axes[1].plot(self.df.iloc[:(len(self.df) - len(pred_df_trf)), 0], label='actual', color='k', **defaultKwargs)
        axes[1].plot(pred_df_trf['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
        axes[1].set_title('Prediction')
        axes[1].set_ylabel('CPU usage [MHz]')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'gt_vs_pred_dots_full')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Lines and dots
        defaultKwargs = {'marker': 'o',
                         'linestyle': '-',
                         'alpha': 0.6,
                         'markersize': 2}
        kwargs_forecast = {'marker': 'o',
                           'linestyle': '-',
                           'alpha': 0.6,
                           'markersize': 2,
                           'color': 'tab:orange'}
        # Construct a figure for the original and new frames.
        fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True, sharey=True)
        plt.suptitle('Test set: GT vs prediction', fontsize=16)
        # Ground Truth
        axes[0].plot(self.df['CPU usage [MHZ]'], label='actual', color='k', **defaultKwargs)
        axes[0].set_title('Ground truth')
        axes[0].set_ylabel('CPU usage [MHz]')
        # Prediction
        axes[1].plot(self.df.iloc[:(len(self.df) - len(pred_df_trf)), 0], label='actual', color='k', **defaultKwargs)
        axes[1].plot(pred_df_trf['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
        axes[1].set_title('Prediction')
        axes[1].set_ylabel('CPU usage [MHz]')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'gt_vs_pred_lines_full')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Save the prediction
        try:
            filename = os.path.join('./logs/ConvLSTM', self.model_name, self.name)
            if not os.access(filename, os.F_OK):
                os.makedirs(filename)
            filename = os.path.join('./logs/ConvLSTM', self.model_name, self.name, 'pred.csv')
            pred_df_trf.to_csv(filename)
        except:
            print("Unable to write to file")
        # Test set
        try:
            filename = os.path.join('./logs/ConvLSTM', self.model_name, self.name, 'test.csv')
            test_df_trf.to_csv(filename)
        except:
            print("Unable to write to file")

        if self.numeric is False:
            return pred, img_pred_bin, pred_df_trf
        else:
            return pred, pred_df_trf

    def evaluation(self, pred, scaler):
        test_trf = scaler.inverse_transform(self.test_df)
        train_trf = scaler.inverse_transform(self.train_df)
        y_true = np.array(test_trf[:len(pred), 0])
        y_pred = np.array(pred['CPU usage [MHZ]'])
        y_train = np.array(train_trf[:, 0])
        pred_trf = pred.copy()
        pred_trf.loc[:, pred_trf.columns] = scaler.transform(pred_trf.loc[:, pred_trf.columns])
        img_pred = self.create_image_numpy(pred_trf.iloc[:, 0], len(pred))
        img_gt = self.create_image_numpy(self.test_df.iloc[:len(pred_trf), 0], len(pred_trf))

        metrics_dic = {'MAE': np.array(tf.keras.metrics.mean_absolute_error(y_true, y_pred)),
                       'MAPE': np.array(tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)),
                       'RMSE': np.sqrt(np.array(tf.keras.metrics.mean_squared_error(y_true, y_pred))),
                       'MASE': mase(y_true, y_pred, y_train),
                       'train_time [s]': self.train_time,
                       'inference_time [s]': self.inference_time,
                       'model_size [B]': self.model_size,
                       'IoU': columnIoU(img_pred, img_gt, 1),
                       'DTW': dtw(y_true, y_pred),
                       'forecasting horizon': self.label_width,
                       }

        # Save metrics
        metrics = pd.DataFrame.from_dict(metrics_dic, orient='index')
        print(metrics)
        try:
            filename = os.path.join('./logs/ConvLSTM', self.model_name, self.name, 'metrics.txt')
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
                      # 'RMSE': rmse_array(y_true, y_pred),
                      }
        errors = pd.DataFrame.from_dict(errors_dic, orient='index')
        errors = errors.T
        print(errors.describe())
        ax = sns.boxplot(data=errors).set(title='Errors box-plot',
                                          ylabel='Error')
        if not os.access(os.path.join(FIGURES_PATH, self.model_name, self.name), os.F_OK):
            os.mkdir(os.path.join(FIGURES_PATH, self.model_name, self.name))
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'errors_boxplot')
        plt.savefig(save_path, bbox_inches='tight')
        # Save the errors
        try:
            filename = os.path.join('./logs/ConvLSTM', self.model_name, self.name, 'errors.csv')
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
            # Adapt to ratio > 1 (see meaning of ratio in init)
            if self.ratio == 1:
                img_pred.append(pred[i, :, -self.label_width:])
            else:
                img_pred.append(pred[i, :, -int(self.label_width / self.ratio):])
        img_pred = np.concatenate(img_pred, axis=1)
        return img_pred

    def image2series(self, img, scaler):
        # Take the brightest pixel of each column
        idx = np.argmax(img, axis=0)
        # Numeric (undo preprocessing)
        pred_numeric = 100 - 1 - idx
        pred_numeric = pred_numeric / 100

        # Do the mapping if ratio > 1 (see meaning of ratio in init)
        if self.ratio != 1:
            pred_numeric = self.mapping(pred_numeric)

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

    def mapping(self, pred_numeric):
        pred_map = []
        for i in range(len(pred_numeric)):
            # Repeat the value ratio times to map it
            for _ in range(self.ratio):
                pred_map.append(pred_numeric[i])
        pred_map = np.array(pred_map)
        return pred_map


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


def DiceLoss(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice = (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return 1 - dice


def DiceBCELoss(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    intersection = K.sum(K.dot(y_true, y_pred))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


def rescale_int(image: np.ndarray):
    image = rescale_intensity(image, in_range=(0, 255))
    return (image * 255).astype("uint8")


def synthetic_dataset(df, freq):
    t = np.arange(0, len(df))
    amplitude = np.sin(2 * np.pi * freq * t)
    df['CPU usage [MHZ]'] = amplitude
    return df


def synthetic_dataset_black(df, freq):
    time = np.arange(0, len(df))
    amplitude = np.sin(2 * np.pi * freq * time)
    amplitude[0] = -5
    amplitude[100] = 5
    df['CPU usage [MHZ]'] = amplitude
    return df
