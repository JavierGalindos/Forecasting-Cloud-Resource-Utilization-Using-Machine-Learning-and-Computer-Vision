import argparse
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, Theta
from darts.metrics import mse, mape, rmse, mase, smape, mae
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from DataExploration.BitbrainsUtils import load_VM
FIGURES_PATH = '../Figures/Modeling'
from darts import TimeSeries


import sys
import time
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


parser = argparse.ArgumentParser(
    description="Training LSTM model (Darts framework)")

parser.add_argument('-e', '--epoch', default=20,
                    help='number of epoch')

parser.add_argument('-n', '--name', default='LSTM',
                    help='name of the model')

parser.add_argument('-s', '--figure_path', default="LSTM",
                    help='Destination for figures to be saved at')

parser.add_argument('-i', '--input_length', default=288,
                    help='input_chunk_length ')

parser.add_argument('-d', '--dim', default=20,
                    help='number of neurons per layer')

parser.add_argument('-l', '--layers', default=1,
                    help='number of rnn layers')

parser.add_argument('--dropout', default=0,
                    help='dropout')

parser.add_argument('--lr', default=1e-3,
                    help='learning rate')

parser.add_argument('--fh', default=288*3,
                    help='forecasting horizon')

args = parser.parse_args()

EPOCH = int(args.epoch)
NAME = args.name
INPUT_CHUNK_LENGTH = int(args.input_length)
FORECASTING_HORIZON = int(args.fh)
HIDDEN_DIM = int(args.dim)
N_LAYERS = int(args.layers)
DROPOUT = float(args.dropout)
LR = float(args.lr)
figure_path = args.figure_path

# Check figure path
figures_path = os.path.join(FIGURES_PATH, figure_path)
if not os.access(figures_path, os.F_OK):
    os.mkdir(figures_path)
if not os.access(figures_path, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(figures_path))
    exit()
else:
    print('figures saved to {}'.format(figures_path))

def eval_model(model):
    pred_series = model.predict(n=FORECASTING_HORIZON, future_covariates=covariates)
    fig = plt.figure(figsize=(8, 5), dpi=150)
    ts_trf.plot(label="actual")
    pred_series.plot(label="forecast")
    plt.title("MAPE: {:.2f}%".format(mape(pred_series, val_trf)))
    plt.legend()
    # Saving figure
    save_path = os.path.join(FIGURES_PATH, figure_path, NAME)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Metrics
    res_mape = mape(val_trf, pred_series)
    res_mae = mae(val_trf, pred_series)
    res_mse = mse(val_trf, pred_series)
    res_rmse = rmse(val_trf, pred_series)
    res_smape = smape(val_trf, pred_series)
    res_mase = mase(val_trf, pred_series,train_trf)
    res_accuracy = {"MAPE":res_mape, "MASE":res_mase, "MAE":res_mae, "RMSE":res_rmse, "MSE":res_mse,"SMAPE": res_smape, "time":res_time}

    results = [pred_series, res_accuracy]
    return results

if __name__ == "__main__":
    # Load data
    VM = load_VM('917.csv')
    series = TimeSeries.from_dataframe(VM, value_cols='CPU usage [MHZ]', freq='5T')
    ts = series

    # replace 0 and NaN by backfilling
    s = series.pd_series()
    s.replace(0.0, np.nan, inplace=True)
    s = s.fillna(method="bfill")
    series = series.from_series(s)
    ts = series

    # split training vs test dataset
    FC_SPLIT = 0.9  # period at which to split training and validation dataset
    train, val = ts.split_after(FC_SPLIT)

    # normalize the time series
    trf = Scaler()
    # fit the transformer to the training dataset
    train_trf = trf.fit_transform(train)
    # apply the transformer to the validation set and the complete series
    val_trf = trf.transform(val)
    ts_trf = trf.transform(ts)

    # create day and hours covariate series
    day_series = datetime_attribute_timeseries(
        pd.date_range(start=series.start_time(),
                      freq=ts.freq_str,
                      periods=len(ts)),
        attribute='day',
        one_hot=False)
    day_series = Scaler().fit_transform(day_series)

    hour_series = datetime_attribute_timeseries(
        day_series,
        attribute='hour',
        one_hot=True)

    covariates = day_series.stack(hour_series)
    cov_train, cov_val = covariates.split_after(FC_SPLIT)

    MODEL = 'LSTM'

    # set the model up
    LTSM_model = RNNModel(
        model=MODEL,
        model_name=MODEL + NAME,
        input_chunk_length=INPUT_CHUNK_LENGTH,
        training_length=INPUT_CHUNK_LENGTH,
        hidden_dim=HIDDEN_DIM,
        n_rnn_layers=N_LAYERS,
        batch_size=16,
        n_epochs=EPOCH,
        dropout=DROPOUT,
        optimizer_kwargs={'lr': LR},
        nr_epochs_val_period=1,
        log_tensorboard=True,
        random_state=42,
        force_reset=True,
        save_checkpoints=True)
    # Fit the model
    t_start = time.perf_counter()
    print("\nBeginning the training")
    res = LTSM_model.fit(
        train_trf,
        future_covariates=covariates,
        val_series=val_trf,
        val_future_covariates=covariates,
        verbose=True)
    res_time = time.perf_counter() - t_start
    print("Training has completed:", f'{res_time:.2f} sec')

    # Take best model
    best_model = RNNModel.load_from_checkpoint(model_name=MODEL + NAME, best=True)
    model_predictions = eval_model(best_model)
    metrics = pd.DataFrame.from_dict(model_predictions[1], orient='index')
    try:
        filename = os.path.join('darts_logs', str(MODEL + NAME), 'metrics.txt')
        file = open(filename, 'wt')
        file.write(str(metrics))
        file.close()

    except:
        print("Unable to write to file")