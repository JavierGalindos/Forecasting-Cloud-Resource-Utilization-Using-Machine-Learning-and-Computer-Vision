#! /bin/bash
python3 LSTM_train.py -e 100 -n 'neurons_20' -d 20 -s 'layers'
python3 LSTM_train.py -e 100 -n 'neurons_50' -d 50 -s 'neurons'
python3 LSTM_train.py -e 100 -n 'neurons_10' -d 10 -s 'neurons'
python3 LSTM_train.py -e 100 -n 'neurons_5' -d 5 -s 'neurons'