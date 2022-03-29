import tensorflow as tf
from ConvLSTM import *


def get_model(ConvLSTMModel, name):
    tf.config.set_soft_device_placement(True)
    if name == "frame":
        """
        Model used in Keras tutorial for next video-frame prediction
        https://keras.io/examples/vision/conv_lstm/
        """
        # Construct the input layer with no definite frame size.
        # inp = tf.keras.layers.Input(shape=(None, *self.train[0].shape[2:]))
        inp = tf.keras.layers.Input(shape=ConvLSTMModel.train[0].shape[1:])

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
        if ConvLSTMModel.numeric is False:
            x = tf.keras.layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same",
            )(x)
        else:
            x = tf.keras.layers.Flatten()(x)
            if ConvLSTMModel.label_width == 1:
                # Shape => [batch, time, features]
                x = tf.keras.layers.Dense(units=1)(x)
            else:
                # Shape => [batch, 1, out_steps*features]
                x = tf.keras.layers.Dense(units=ConvLSTMModel.label_width, kernel_initializer=tf.initializers.zeros())(
                    x)
                # Shape => [batch, out_steps, features]
                x = tf.keras.layers.Reshape([ConvLSTMModel.label_width, 1])(x)

        # Next, we will build the complete model and compile it.
        model = tf.keras.models.Model(inp, x)
        if ConvLSTMModel.numeric is False:
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=tf.metrics.BinaryAccuracy(),
            )
        else:
            model.compile(loss=tf.losses.MeanSquaredError(),
                          optimizer=tf.optimizers.Adam(),
                          metrics=tf.metrics.MeanAbsoluteError(),
                          )
    elif name == "LRCN":
        """
        Long Recurrent Convolution Neural Network (CNN + RNN)
        https://colab.research.google.com/drive/1RtTYonaJ7ASX_ZMzcV3t_0jNktheKQF9?usp=sharing#scrollTo=X8nAG3xYA5lW
        """

        setattr(ConvLSTMModel, 'numeric', True)
        # We will use a Sequential model for model construction.
        model = tf.keras.models.Sequential()

        # Define the Model Architecture.

        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                                                  input_shape=ConvLSTMModel.train[0].shape[1:]))

        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((4, 4))))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.25)))

        model.add(
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((4, 4))))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.25)))

        model.add(
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.25)))

        model.add(
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))))
        # model.add(TimeDistributed(Dropout(0.25)))

        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

        model.add(tf.keras.layers.LSTM(32))

        if ConvLSTMModel.label_width == 1:
            # Shape => [batch, time, features]
            model.add(tf.keras.layers.Dense(units=1))
        else:
            # Shape => [batch, 1, out_steps*features]
            model.add(
                tf.keras.layers.Dense(units=ConvLSTMModel.label_width, kernel_initializer=tf.initializers.zeros()))
        # Shape => [batch, out_steps, features]
        model.add(tf.keras.layers.Reshape([ConvLSTMModel.label_width, 1]))

        # Compile
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=tf.metrics.MeanAbsoluteError(),
                      )
    elif name == "ConvLSTM":
        """
        ConvLSTM with down sampling
        https://colab.research.google.com/drive/1RtTYonaJ7ASX_ZMzcV3t_0jNktheKQF9?usp=sharing#scrollTo=X8nAG3xYA5lW
        """
        setattr(ConvLSTMModel, 'numeric', True)
        # We will use a Sequential model for model construction.
        model = tf.keras.models.Sequential()

        # Define the Model Architecture.

        model.add(
            tf.keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                       recurrent_dropout=0.2, return_sequences=True,
                                       input_shape=ConvLSTMModel.train[0].shape[1:]))

        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))

        model.add(
            tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                       recurrent_dropout=0.2, return_sequences=True))

        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))

        model.add(
            tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                       recurrent_dropout=0.2, return_sequences=True))

        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))

        model.add(
            tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                       recurrent_dropout=0.2, return_sequences=True))

        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        # model.add(TimeDistributed(Dropout(0.2)))

        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

        model.add(tf.keras.layers.LSTM(32))

        if ConvLSTMModel.label_width == 1:
            # Shape => [batch, time, features]
            model.add(tf.keras.layers.Dense(units=1))
        else:
            # Shape => [batch, 1, out_steps*features]
            model.add(
                tf.keras.layers.Dense(units=ConvLSTMModel.label_width, kernel_initializer=tf.initializers.zeros()))
        # Shape => [batch, out_steps, features]
        model.add(tf.keras.layers.Reshape([ConvLSTMModel.label_width, 1]))

        # Compile
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=tf.metrics.MeanAbsoluteError(),
                      )
    else:
        raise KeyError("{} model is unknown.".format(name))

    return model
