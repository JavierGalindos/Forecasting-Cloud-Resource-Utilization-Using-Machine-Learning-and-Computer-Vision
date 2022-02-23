# Modeling

## LSTM
### train_LSTM.py
Trains an LSTM model over the Darts framework.

To run:
```
python train_LSTM.py -e <number of epoch> -n <name of the model> -s <path to save the figure> -i <input_chunck_length> -d <number of neurons> -l <number or rnn layers --dropout <droput rate> --lr <learning rate> --fh <forecasting horizon>
```

Example:
```
python3 train_LSTM.py -e 20 -n 'LSTM_input_1day' -i 288 -s 'LSTM/input_chunk'
```