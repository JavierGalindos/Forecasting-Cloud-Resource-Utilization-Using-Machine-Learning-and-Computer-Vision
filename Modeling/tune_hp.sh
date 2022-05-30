#! /bin/bash
python3 ConvLSTM_CM_allVM.py -m 'video' -n '917/CM917/1' -i 128 --VM 917 --label 16
python3 LSTM_CM_allVM.py -n '917/CM917/1' -i 128 --VM 917 --label 16
python3 ConvLSTM_CM_allVM.py -m 'video' -n '917/CM118/1' -i 128 --VM 118 --label 16
python3 LSTM_CM_allVM.py -n '917/CM118/1' -i 128 --VM 118 --label 16
python3 ConvLSTM_CM_allVM.py -m 'video' -n '917/CM226/1' -i 128 --VM 226 --label 16
python3 LSTM_CM_allVM.py -n '917/CM226/1' -i 128 --VM 226 --label 16



