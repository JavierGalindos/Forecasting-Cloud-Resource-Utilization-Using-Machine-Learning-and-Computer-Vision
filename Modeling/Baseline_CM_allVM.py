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

# parser.add_argument('--VM_test', default=340,
#                     help='VM number')

args = parser.parse_args()

MODEL_NAME = args.model_name
NAME = args.name
LABEL_LENGTH = int(args.label)
VM_NUM = int(args.VM)
# VM_NUM_test = int(args.VM_test)

if __name__ == "__main__":

    # Pre-processing for cross-modeling
    print('Pre-processing')
    # Load data
    VM = load_VM(f'{VM_NUM}.csv')
    # Make it univariate
    df = VM[['CPU usage [MHZ]']]

    # Make data sythetic
    # df = synthetic_dataset(df, 1/25)

    # Split the data
    # (80%, 20%, 0%) split for the training, validation from one VM
    train_df, val_df, _ = split_data(df, 0.8, 0.19)
    # Test set from other VM
    VM_NUM_test = 1
    VM_test = load_VM(f'{VM_NUM_test}.csv')
    test_df = VM_test[['CPU usage [MHZ]']]

    # Naive forecast
    naive_model = Baseline(label_width=LABEL_LENGTH,
                           df=df,
                           train_df=train_df,
                           val_df=val_df,
                           test_df=test_df,
                           model_name='naive',
                           name=NAME)

    for VM in range(1, 1250):
        VM_test = load_VM(f'{VM}.csv')
        df_test = VM_test[['CPU usage [MHZ]']]
        # Test
        scaler = MinMaxScaler()
        df_test.loc[:, df_test.columns] = scaler.fit_transform(df_test.loc[:, df_test.columns])
        test_df = df_test.copy()

        # Change class attributes
        setattr(naive_model, 'test_df', test_df)
        setattr(naive_model, 'name', f'917/CM_{VM_NUM}/{VM}')

        # Prediction
        try:
            print('Prediction:')
            pred_df = naive_model.baseline_prediction()
            # Evaluation
            print('Evaluation:')
            metrics = naive_model.baseline_evaluate(pred_df)
        except:
            print(f"VM{VM} failed")
