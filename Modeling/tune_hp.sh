#! /bin/bash
python3 ConvLSTM_CM_allVM.py -m 'video' -n '917/CM_917/1' -i 128 --VM 917 --label 16
python3 ConvLSTM_CM_allVM.py -m 'video' -n '118/CM_118/1' -i 128 --VM 118 --label 16
python3 ConvLSTM_CM_allVM.py -m 'video' -n '226/CM_226/1' -i 128 --VM 226 --label 16
python3 LSTM_CM_allVM.py -n '917/CM917/1' -i 128 --VM 917 --label 16
python3 LSTM_CM_allVM.py -n '118/CM118/1' -i 128 --VM 118 --label 16
python3 LSTM_CM_allVM.py -n '226/CM226/1' -i 128 --VM 226 --label 16

python3 LSTM_train.py -n '917/labels_16' --label 16
python3 LSTM_train.py -n '917/labels_16_clas' --label 16 --classification True
python3 ConvLSTM_train.py -m 'video' -n '917/labels_16' --label 16




