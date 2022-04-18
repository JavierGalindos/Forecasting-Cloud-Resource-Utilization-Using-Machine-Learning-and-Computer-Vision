#! /bin/bash
python3 ConvLSTM_train.py -m 'AE' -n '917/overlap_50' -i 32 --VM 917 --overlapping 0.50 --label 16
python3 ConvLSTM_train.py -m 'AE' -n '917/overlap_75' -i 64 --VM 917 --overlapping 0.75 --label 16
python3 ConvLSTM_train.py -m 'AE' -n '917/overlap_80' -i 80 --VM 917 --overlapping 0.80 --label 16
python3 ConvLSTM_train.py -m 'AE' -n '917/overlap_90' -i 160 --VM 917 --overlapping 0.90 --label 16
python3 ConvLSTM_train.py -m 'AE' -n '917/overlap_95' -i 320 --VM 917 --overlapping 0.95 --label 16
python3 ConvLSTM_train.py -m 'LRCN' -n '917/overlap_50' -i 32 --VM 917 --overlapping 0.50 --label 16
python3 ConvLSTM_train.py -m 'LRCN' -n '917/overlap_75' -i 64 --VM 917 --overlapping 0.75 --label 16
python3 ConvLSTM_train.py -m 'LRCN' -n '917/overlap_80' -i 80 --VM 917 --overlapping 0.80 --label 16
python3 ConvLSTM_train.py -m 'LRCN' -n '917/overlap_90' -i 160 --VM 917 --overlapping 0.90 --label 16
python3 ConvLSTM_train.py -m 'LRCN' -n '917/overlap_95' -i 320 --VM 917 --overlapping 0.95 --label 16
python3 ConvLSTM_train.py -m 'video' -n '917/overlap_50' -i 32 --VM 917 --overlapping 0.50 --label 16
python3 ConvLSTM_train.py -m 'video' -n '917/overlap_75' -i 64 --VM 917 --overlapping 0.75 --label 16
python3 ConvLSTM_train.py -m 'video' -n '917/overlap_80' -i 80 --VM 917 --overlapping 0.80 --label 16
python3 ConvLSTM_train.py -m 'video' -n '917/overlap_90' -i 160 --VM 917 --overlapping 0.90 --label 16
python3 ConvLSTM_train.py -m 'video' -n '917/overlap_95' -i 320 --VM 917 --overlapping 0.95 --label 16




