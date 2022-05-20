#! /bin/bash
python3 Baseline_train.py -n '917/labels_32' --label 32
python3 LSTM_train.py -n '917/labels_32' --label 32
#python3 ConvLSTM_train.py -m 'video' -n '917/labels_32' --label 32
#python3 ConvLSTM_train.py -m 'LRCN' -n '917/labels_32' --label 32
#python3 ConvLSTM_CM.py -m 'video' -n '917/CM_119' -i 128 --VM 917 --label 16 --VM_test 119
python3 LSTM_CM.py  -n '917/CM_119' --VM 917 --label 16 --VM_test 119
#python3 ConvLSTM_CM.py -m 'video' -n '917/CM_340' -i 128 --VM 917 --label 16 --VM_test 340
python3 LSTM_CM.py  -n '917/CM_340' --VM 917 --label 16 --VM_test 340
#python3 ConvLSTM_CM.py -m 'video' -n '917/CM_322' -i 128 --VM 917 --label 16 --VM_test 322
python3 LSTM_CM.py  -n '917/CM_322' --VM 917 --label 16 --VM_test 322




