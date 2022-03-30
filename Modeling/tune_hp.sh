#! /bin/bash
python3 ConvLSTM_train.py -m 'LRCN' -n '917/frames_1' -i 64 --label 16 --frames 1
python3 ConvLSTM_train.py -m 'LRCN' -n '917/frames_4' -i 64 --label 16 --frames 4
python3 ConvLSTM_train.py -m 'LRCN' -n '917/frames_8' -i 64 --label 16 --frames 8
python3 ConvLSTM_train.py -m 'LRCN' -n '917/frames_16' -i 64 --label 16 --frames 16



