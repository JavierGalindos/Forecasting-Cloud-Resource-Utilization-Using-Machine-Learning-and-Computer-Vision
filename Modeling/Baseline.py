# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
# from itertools import product
from DataExploration.BitbrainsUtils import *
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from DataExploration.BitbrainsUtils import mase
import tensorflow as tf
import time

FIGURES_PATH = '../Figures/Modeling/Baseline'

if not os.access(FIGURES_PATH, os.F_OK):
    os.makedirs(FIGURES_PATH)
if not os.access(FIGURES_PATH, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(FIGURES_PATH))
    exit()
else:
    print('figures saved to {}'.format(FIGURES_PATH))


class Baseline:
    def __init__(self, label_width, df,
                 train_df, val_df, test_df,
                 model_name='ARIMA', name='917',
                 ):
        # Store the raw data.
        self.df = df
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Window parameters.
        self.label_width = label_width
        self.name = name
        self.model_name = model_name

    def arima_fit_predict(self):
        t_start = time.perf_counter()
        self.model = auto_arima(self.train_df, start_p=1, start_q=1,
                                max_p=5, max_q=5,
                                d=0,
                                seasonal=False,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True,
                                n_jobs=-1)
        print(self.model.summary())
        self.train_time = time.perf_counter() - t_start

        # Prediction
        t_start = time.perf_counter()
        pred = []
        n = len(self.test_df)
        for i in tqdm(range(1, n, self.label_width)):
            test_aux = self.test_df.iloc[:i, :]
            self.model.update(test_aux)
            pred.append(self.model.predict(n_periods=self.label_width))
        self.inference_time = time.perf_counter() - t_start
        pred = np.reshape(pred, (-1, 1))
        pred = pred[:len(self.test_df), :]
        self.test_df = self.test_df.iloc[:len(pred), :]
        pred_df = pd.DataFrame(data=np.array(pred), columns=['CPU usage [MHZ]'], index=self.test_df.index)
        return pred_df

    def expSmooth_fit_pred(self):
        t_start = time.perf_counter()
        self.model = SimpleExpSmoothing
        pred = []
        for i in tqdm(range(0, len(self.test_df), self.label_width)):
            training = pd.concat([self.train_df, self.test_df.iloc[:i, :]])
            fit = self.model(training).fit()
            pred.append(fit.forecast(self.label_width))
        self.train_time = time.perf_counter() - t_start
        self.inference_time = self.train_time
        pred = np.reshape(pred, (-1, 1))
        pred = pred[:len(self.test_df), :]
        pred_df = pd.DataFrame(data=np.array(pred), columns=['CPU usage [MHZ]'], index=self.test_df.index)
        return pred_df

    def baseline_prediction(self):
        if self.model_name == "ARIMA":
            pred_df = self.arima_fit_predict()
        elif self.model_name == "exp":
            pred_df = self.expSmooth_fit_pred()
        else:
            raise KeyError("{} model is unknown.".format(self.model_name))

        # Figures
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
        fig = plt.figure(dpi=200, figsize=(20, 5))
        plt.grid()
        self.df['CPU usage [MHZ]'].plot(label='actual', color='k', **defaultKwargs)
        pred_df['CPU usage [MHZ]'].plot(label='forecast', **kwargs_forecast)
        plt.ylabel('CPU usage [MHz]')
        # plt.title()
        plt.grid()
        plt.legend()
        if not os.access(os.path.join(FIGURES_PATH, self.model_name, self.name), os.F_OK):
            os.makedirs(os.path.join(FIGURES_PATH, self.model_name, self.name))
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'forecast')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # Figure zoom
        fig = plt.figure(dpi=200, figsize=(20, 5))
        plt.grid()
        self.test_df['CPU usage [MHZ]'].plot(label='actual', color='k', **defaultKwargs)
        pred_df['CPU usage [MHZ]'].plot(label='forecast', **kwargs_forecast)
        plt.ylabel('CPU usage [MHz]')
        # plt.title()
        plt.grid()
        plt.legend()
        if not os.access(os.path.join(FIGURES_PATH, self.model_name, self.name), os.F_OK):
            os.makedirs(os.path.join(FIGURES_PATH, self.model_name, self.name))
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
        axes[0].plot(self.test_df['CPU usage [MHZ]'], label='actual', color='k', **defaultKwargs)
        axes[0].set_title('Ground truth')
        axes[0].set_ylabel('CPU usage [MHz]')
        # Prediction
        axes[1].plot(pred_df['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
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
        axes[0].plot(self.test_df['CPU usage [MHZ]'], label='actual', color='k', **defaultKwargs)
        axes[0].set_title('Ground truth')
        axes[0].set_ylabel('CPU usage [MHz]')
        # Prediction
        axes[1].plot(pred_df['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
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
        axes[1].plot(self.df.iloc[:(len(self.df) - len(pred_df)), 0], label='actual', color='k', **defaultKwargs)
        axes[1].plot(pred_df['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
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
        axes[1].plot(self.df.iloc[:(len(self.df) - len(pred_df)), 0], label='actual', color='k', **defaultKwargs)
        axes[1].plot(pred_df['CPU usage [MHZ]'], label='forecast', **kwargs_forecast)
        axes[1].set_title('Prediction')
        axes[1].set_ylabel('CPU usage [MHz]')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        save_path = os.path.join(FIGURES_PATH, self.model_name, self.name, 'gt_vs_pred_lines_full')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return pred_df

    def baseline_evaluate(self, pred):
        y_true = np.array(self.test_df.iloc[:, 0])
        y_pred = np.array(pred['CPU usage [MHZ]'])
        y_train = np.array(self.train_df.iloc[:, 0])
        scaler = MinMaxScaler()
        img_pred = self.create_image_numpy(scaler.fit_transform(np.array(pred.iloc[:, 0]).reshape(-1, 1)), len(pred))
        img_gt = self.create_image_numpy(scaler.fit_transform(np.array(self.test_df.iloc[:len(pred), 0]).reshape(-1, 1)), len(pred))
        metrics_dic = {'MAE': np.array(tf.keras.metrics.mean_absolute_error(y_true, y_pred)),
                       'MAPE': np.array(tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)),
                       'RMSE': np.sqrt(np.array(tf.keras.metrics.mean_squared_error(y_true, y_pred))),
                       'MASE': mase(y_true, y_pred, y_train),
                       'train_time [s]': self.train_time,
                       'inference_time [s]': self.inference_time,
                       'model_size [B]': 'NaN',
                       'IoU': columnIoU(img_pred, img_gt, 1),
                       'DTW': dtw(y_true, y_pred),
                       'forecasting horizon': self.label_width,
                       }

        # Save metrics
        metrics = pd.DataFrame.from_dict(metrics_dic, orient='index')
        print(metrics)
        try:
            filename = os.path.join('logs/Baseline', self.model_name, self.name)
            if not os.access(filename, os.F_OK):
                os.makedirs(filename)
            filename = os.path.join('logs/Baseline', self.model_name, self.name, 'metrics.txt')
            metrics.to_csv(filename)
        except:
            print("Unable to write to file")
        return metrics

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
