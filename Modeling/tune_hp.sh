#! /bin/bash
python3 LSTM_train.py -e 100 -n '917/dropout_0' -s '917/dropout' --label 6 --dropout 0
python3 LSTM_train.py -e 100 -n '917/dropout_01' -s '917/dropout' --label 6 --dropout 0.1
python3 LSTM_train.py -e 100 -n '917/dropout_02' -s '917/dropout' --label 6 --dropout 0.2
python3 LSTM_train.py -e 100 -n '917/dropout_04' -s '917/dropout' --label 6 --dropout 0.4



