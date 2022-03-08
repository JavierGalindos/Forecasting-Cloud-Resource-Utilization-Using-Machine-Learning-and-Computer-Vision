#! /bin/bash
python3 LSTM_train.py -e 100 -n 'neurons_20' -d 20 -s 'neurons' --label 9
python3 LSTM_train.py -e 100 -n 'neurons_5' -d 5 -s 'neurons' --label 9
python3 LSTM_train.py -e 100 -n 'neurons_100' -d 100 -s 'neurons' --label 9
python3 LSTM_train.py -e 100 -n 'neurons_50' -d 50 -s 'neurons' --label 9