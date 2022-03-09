#! /bin/bash
python3 LSTM_train.py -e 100 -n 'layers_1' -l 1 -s 'layers' --label 9
python3 LSTM_train.py -e 100 -n 'layers_2' -l 2 -s 'layers' --label 9
python3 LSTM_train.py -e 100 -n 'layers_3' -l 3 -s 'layers' --label 9