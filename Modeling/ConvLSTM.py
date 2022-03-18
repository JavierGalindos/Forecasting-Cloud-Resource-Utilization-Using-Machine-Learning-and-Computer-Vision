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

from DataExploration.BitbrainsUtils import load_VM, plot_timeSeries, mase, split_data, data_transformation, reg2class, \
    class2num

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

# Global figure size and resolution. Gives image of size 64x64
dpi = 8
figsize = (8, 8)


class ConvLSTMModel:
    # TODO: Clean useless hyperameters and classification code
    def __init__(self, input_width, label_width, df,
                 train_df, val_df, test_df,
                 epoch=100, units=20, layers=1,
                 dropout=0, batch_size=16,
                 name='ConvLSTM', classification=False,
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

        self.model = self.get_model()

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

    def create_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        # Check if the length of series is multiple of label_width
        if ((len(data) - self.input_width) % self.label_width) != 0:
            # If not: trim the series to make it multiple
            data = data[
                   :math.floor((len(data) - self.input_width) / self.label_width) * self.label_width + self.input_width,
                   :]

        # Generate the images
        # Overlapping c=(input_width-label_width)/input_width
        # Input images (shift label with between samples)
        input = []
        labels = []
        for i in range(0, len(data) - self.input_width, self.label_width):
            # Input
            fig = full_frame(figsize, dpi)
            plt.ylim(0, 1)  # set y-axis limits
            plt.plot(data[i:(i + self.input_width), :], ',', color='black')
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
            # Normalize image
            img_normalized = img / 255.
            input.append(img_normalized)

            # Labels
            fig = full_frame(figsize, dpi)
            plt.ylim(0, 1)  # set y-axis limits
            plt.plot(data[(i + self.label_width):(i + self.input_width + self.label_width), :], '-', color='black')
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
            # Normalize image
            img_normalized = img / 255.
            labels.append(img_normalized)

        input = np.array(input)
        input = np.expand_dims(input, axis=(1, 4))
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=(1, 4))

        return input, labels

    def create_image_dataset(self, data, image_path):
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
        inp = tf.keras.layers.Input(shape=self.train[0].shape[1:])

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
        x = tf.keras.layers.Reshape([64, 64, 64])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape([1, 64, 64, 64])(x)
        x = tf.keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            data_format="channels_last",
        )(x)
        x = tf.keras.layers.Reshape([64, 64, 64])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape([1, 64, 64, 64])(x)
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
            loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(),
        )
        return model

    def compile_and_fit(self):
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=15,
                                                          mode='min',
                                                          restore_best_weights=True)

        # Tensorboard
        # log_dir = f'logs/fit/{self.name}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{self.name}/tensorboard'
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
                                 validation_data=(self.val[0], self.val[1]),
                                 callbacks=[early_stopping, tensorboard_callback, reduce_lr])

        self.train_time = time.perf_counter() - t_start
        print("Training has completed:", f'{self.train_time:.2f} sec')

        # Save the best model (getting the best with early_stopping callback)
        # Check the path exists
        model_filepath = f'logs/{self.name}/checkpoints'
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
            os.mkdir(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'Loss')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return history

    def prediction(self):
        print('Inference:')
        t_start = time.perf_counter()
        pred = self.model.predict(self.test_pred[0], batch_size=self.batch_size)
        self.inference_time = time.perf_counter() - t_start
        print("Inference time:", f'{self.inference_time:.2f} sec')

        # Ground truth
        gt = self.test[1]
        # Construct a figure for the original and new frames.
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))

        # Plot the original frames.
        for idx, ax in enumerate(axes[0]):
            ax.imshow(np.squeeze(gt[idx]), cmap="gray")
            ax.set_title(f"Sample {idx}")
            ax.axis("off")

        # Plot the new frames.
        for idx, ax in enumerate(axes[1]):
            ax.imshow(np.squeeze(pred[idx]), cmap="gray")
            ax.set_title(f"Sample {idx}")
            ax.axis("off")

        # Display the figure.
        # plt.show()
        if not os.access(os.path.join(FIGURES_PATH, self.name), os.F_OK):
            os.mkdir(os.path.join(FIGURES_PATH, self.name))
        save_path = os.path.join(FIGURES_PATH, self.name, 'image_example')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        return pred


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

