#! /bin/bash
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_1' -i 64 --label 16 --img_width 64 --VM 917
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_2' -i 128 --label 16 --img_width 64 --VM 917
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_3' -i 192 --label 16 --img_width 64 --VM 917
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_4' -i 256 --label 16 --img_width 64 --VM 917



