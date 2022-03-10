#! /bin/bash
python3 LSTM_train.py -e 100 -n '541/labels_1' -s '541/labels' --label 1
python3 LSTM_train.py -e 100 -n '541/labels_3' -s '541/labels' --label 3
python3 LSTM_train.py -e 100 -n '541/labels_9' -s '541/labels' --label 9
python3 LSTM_train.py -e 100 -n '541/labels_16' -s '541/labels' --label 16
python3 LSTM_train.py -e 100 -n '541/labels_36' -s '541/labels' --label 36
python3 LSTM_train.py -e 100 -n '541/labels_72' -s '541/labels' --label 72