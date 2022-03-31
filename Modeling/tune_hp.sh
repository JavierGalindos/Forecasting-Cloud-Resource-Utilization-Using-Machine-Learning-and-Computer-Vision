#! /bin/bash
python3 ConvLSTM_train.py -m 'video' -n '917/input_16' -i 64 --label 16 -i 16 --frames 8
python3 ConvLSTM_train.py -m 'video' -n '917/input_32' -i 64 --label 16 -i 32 --frames 8
python3 ConvLSTM_train.py -m 'video' -n '917/input_64' -i 64 --label 16 -i 64 --frames 8
python3 ConvLSTM_train.py -m 'video' -n '917/input_128' -i 64 --label 16 -i 128 --frames 8



