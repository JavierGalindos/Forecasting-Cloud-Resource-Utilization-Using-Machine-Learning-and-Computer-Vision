#! /bin/bash
python3 Baseline_train.py -n '917/CM_322' --label 16 --VM_test 322
python3 LSTM_CM.py  -n '917/CM_322' --VM 917 --label 16 --VM_test 322





