#! /bin/bash
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_2' -i 128 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_3' -i 192 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_4' -i 256 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'AE' -n '917/ratio_5' -i 320 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'video' -n '917/ratio_1' -i 128 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'video' -n '917/ratio_2' -i 128 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'video' -n '917/ratio_3' -i 192 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'video' -n '917/ratio_4' -i 256 --label 16 --img_width 64
python3 ConvLSTM_train.py -m 'video' -n '917/ratio_5' -i 320 --label 16 --img_width 64



