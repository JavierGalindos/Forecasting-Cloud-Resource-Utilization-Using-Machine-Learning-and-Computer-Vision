# Importation
import os
import matplotlib.pyplot as plt
import pandas as pd
import LSTM_Prep
from DataExploration.BitbrainsUtils import load_VM

FIGURES_PATH = '../Figures/Modeling/LSTM'

if not os.access(FIGURES_PATH, os.F_OK):
    os.mkdir(FIGURES_PATH)
if not os.access(FIGURES_PATH, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(FIGURES_PATH))
    exit()
else:
    print('figures saved to {}'.format(FIGURES_PATH))


if __name__ == "__main__":
    # Data
    VM = pd.read_csv('../Datasets/fastStorage/2013-8/917.csv', sep=';\t', engine='python')
    dat = VM

    split = 0.8
    sequence_length = 288

    data_prep = LSTM_Prep.Data_Prep(dataset=dat)
    rnn_df, validation_df = data_prep.preprocess_rnn(date_colname='Timestamp [ms]', numeric_colname='CPU usage [MHZ]',
                                                     pred_set_timesteps=288)

    series_prep = LSTM_Prep.Series_Prep(rnn_df=rnn_df, numeric_colname='CPU usage [MHZ]')
    window, X_min, X_max = series_prep.make_window(sequence_length=sequence_length,
                                                   train_test_split=split,
                                                   return_original_x=True)

    X_train, X_test, y_train, y_test = series_prep.reshape_window(window, train_test_split=split)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #                 Building the LSTM
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.callbacks import ReduceLROnPlateau  # Learning rate scheduler for when we reach plateaus

    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)


    # Reset model if we want to re-train with different splits
    def reset_weights(model):
        import keras.backend as K
        session = K.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)


    # Epochs and validation split
    EPOCHS = 201  # 201
    validation = 0.05

    # Instantiate the model
    model = Sequential()

    # Add the first layer.... the input shape is (Sample, seq_len-1, 1)
    model.add(LSTM(
        input_shape=(sequence_length - 1, 1), return_sequences=True,
        units=100))

    # Add the second layer.... the input shape is (Sample, seq_len-1, 1)
    model.add(LSTM(
        input_shape=(sequence_length - 1, 1),
        units=100))

    # Add the output layer, simply one unit
    model.add(Dense(
        units=1,
        activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam')

    # History object for plotting our model loss by epoch
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=validation,
                        callbacks=[rlrop])
    # Loss History
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    save_path = os.path.join(FIGURES_PATH, 'Loss')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #              Predicting the future
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Creating our future object
    future = LSTM_Prep.Predict_Future(X_test=X_test, validation_df=validation_df, lstm_model=model)
    # Checking its accuracy on our training set
    future.predicted_vs_actual(X_min=X_min, X_max=X_max, numeric_colname='CPU usage [MHZ]')
    # Predicting 'x' timesteps out
    future.predict_future(X_min=X_min, X_max=X_max, numeric_colname='CPU usage [MHZ]', timesteps_to_predict=15, return_future=True)
