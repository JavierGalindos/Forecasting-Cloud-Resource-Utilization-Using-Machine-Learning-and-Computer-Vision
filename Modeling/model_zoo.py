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
            tf.keras.layers.ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                       recurrent_dropout=0.2, return_sequences=True,
                                       input_shape=ConvLSTMModel.train[0].shape[1:]))

        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))

        model.add(
            tf.keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                       recurrent_dropout=0.2, return_sequences=True))

        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))

        model.add(
            tf.keras.layers.ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                       recurrent_dropout=0.2, return_sequences=True))

        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))

        model.add(
            tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
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

    elif name == "video":
        """
        Video frame prediction
        https://github.com/lukas/ml-class/tree/master/videos/video-predict
        """
        setattr(ConvLSTMModel, 'numeric', False)
        inp = tf.keras.layers.Input(shape=ConvLSTMModel.train[0].shape[1:])
        # Conv2DLSTM
        c = 32

        x = (tf.keras.layers.ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', name='conv_lstm1',
                                        return_sequences=True))(
            inp)

        c1 = (tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = (tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))(c1)

        x = (tf.keras.layers.ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same', name='conv_lstm3',
                                        return_sequences=True))(x)
        c2 = (tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = (tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))(c2)
        x = (tf.keras.layers.ConvLSTM2D(filters=4 * c, kernel_size=(3, 3), padding='same', name='conv_lstm4',
                                        return_sequences=True))(x)

        x = (tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2))))(x)
        x = (tf.keras.layers.ConvLSTM2D(filters=4 * c, kernel_size=(3, 3), padding='same', name='conv_lstm5',
                                        return_sequences=True))(x)
        x = (tf.keras.layers.BatchNormalization())(x)

        x = (tf.keras.layers.ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same', name='conv_lstm6',
                                        return_sequences=True))(x)
        x = (tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.Add()([c2, x])
        x = tf.keras.layers.Dropout(0.2)(x)

        x = (tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2))))(x)
        x = (tf.keras.layers.ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same', name='conv_lstm7',
                                        return_sequences=False))(
            x)
        x = (tf.keras.layers.BatchNormalization())(x)
        #         combined = concatenate([last_layer*0.20, x])
        combined = x
        combined = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(combined)
        combined = tf.expand_dims(combined, axis=1)
        model = tf.keras.models.Model(inputs=[inp], outputs=[combined])

        # Compile
        model.compile(loss=tf.losses.BinaryCrossentropy(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=tf.metrics.BinaryAccuracy(),
                      )

    elif name == "AE":
        """
        JP Morgan paper: Visual Time Series Forecasting: An Image-driven Approach
        https://arxiv.org/abs/2107.01273
        """
        latent_dim = 64
        setattr(ConvLSTMModel, 'numeric', False)
        setattr(ConvLSTMModel, 'frames', 1)
        inp = tf.keras.layers.Input(shape=ConvLSTMModel.train[0].shape[1:])

        x = tf.keras.layers.Reshape(target_shape=ConvLSTMModel.train[0].shape[2:])(inp)
        x = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same', strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same', strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        # x = tf.keras.layers.Conv2D(512, (5, 5), activation='relu', padding='same', strides=2)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        encoded = tf.keras.layers.Dense(latent_dim)(x)

        # at this point the representation is latent_dim-dimensional

        # Units = width*length*filters
        x = tf.keras.layers.Dense((ConvLSTMModel.train[0].shape[2]/4)*(ConvLSTMModel.train[0].shape[3]/4)*256, activation='relu')(encoded)
        x = tf.keras.layers.Reshape([int(ConvLSTMModel.train[0].shape[2]/4), int(ConvLSTMModel.train[0].shape[3]/4), 256])(x)
        # x = tf.keras.layers.Conv2DTranspose(512, (5, 5), activation='relu', padding='same', strides=2)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(256, (5, 5), activation='relu', padding='same', strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(128, (5, 5), activation='relu', padding='same', strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        decoded = tf.keras.layers.Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)
        decoded = tf.keras.layers.Reshape(target_shape=ConvLSTMModel.train[1].shape[1:])(decoded)

        model = tf.keras.Model(inp, decoded)

        # Compile
        model.compile(loss=tf.losses.BinaryCrossentropy(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=tf.metrics.BinaryAccuracy(),
                      )

    else:
        raise KeyError("{} model is unknown.".format(name))

    return model
