import argparse
from Baseline import *

parser = argparse.ArgumentParser(
    description="Training Baseline")

parser.add_argument('-m', '--model_name', default='ARIMA',
                    help='model_name structure')

parser.add_argument('-n', '--name', default='917/labels_1',
                    help='name of the model')

parser.add_argument('--label', default=16,
                    help='label_length')

parser.add_argument('--VM', default=917,
                    help='VM number')

args = parser.parse_args()


MODEL_NAME = args.model_name
NAME = args.name
LABEL_LENGTH = int(args.label)
VM_NUM = int(args.VM)

if __name__ == "__main__":
    print('Pre-processing')
    # Load data
    VM = load_VM(f'{VM_NUM}.csv')
    # VM = load_VM('541.csv')
    # Make it univariate
    df = VM[['CPU usage [MHZ]']]

    # Split data
    train_df, val_df, test_df = split_data(df, 0.9, 0)

    # Exponential smoothing
    exp_model = Baseline(label_width=LABEL_LENGTH,
                         df=df,
                         train_df=train_df,
                         val_df=val_df,
                         test_df=test_df,
                         model_name='exp',
                         name=NAME)
    # Prediction
    pred_df = exp_model.baseline_prediction()
    # Validation
    metrics = exp_model.baseline_evaluate(pred_df)

    # ARIMA
    arima_model = Baseline(label_width=LABEL_LENGTH,
                           df=df,
                           train_df=train_df,
                           val_df=val_df,
                           test_df=test_df,
                           model_name='ARIMA',
                           name=NAME)
    # Prediction
    pred_df = arima_model.baseline_prediction()
    # Validation
    metrics = arima_model.baseline_evaluate(pred_df)


