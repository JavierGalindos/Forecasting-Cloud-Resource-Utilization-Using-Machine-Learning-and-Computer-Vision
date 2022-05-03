#! /bin/bash
python3 ConvLSTM_train.py -m 'AE' -n '917/overlap_0' -i 16 --VM 917 --overlapping 0 --label 16
python3 ConvLSTM_train.py -m 'video' -n '917/overlap_0' -i 16 --VM 917 --overlapping 0 --label 16
python3 ConvLSTM_train.py -m 'LRCN' -n '917/overlap_0' -i 16 --VM 917 --overlapping 0 --label 16





