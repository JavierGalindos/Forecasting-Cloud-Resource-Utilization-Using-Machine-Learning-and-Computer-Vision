#! /bin/bash
python3 ConvLSTM_train.py -m 'LRCN' -n '917/input_64' -i 64 --label 16 --img_width 64 --VM 917
python3 ConvLSTM_train.py -m 'LRCN' -n '917/input_128' -i 128 --label 16 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'LRCN' -n '917/input_256' -i 256 --label 16 --img_width 256 --VM 917
python3 ConvLSTM_train.py -m 'video' -n '917/input_64' -i 64 --label 16 --img_width 64 --VM 917
python3 ConvLSTM_train.py -m 'video' -n '917/input_128' -i 128 --label 16 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'video' -n '917/input_256' -i 256 --label 16 --img_width 256 --VM 917
python3 ConvLSTM_train.py -m 'AE' -n '917/labels_1' -i 128 --label 1 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'AE' -n '917/labels_8' -i 128 --label 8 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'AE' -n '917/labels_16' -i 128 --label 16 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'AE' -n '917/labels_32' -i 128 --label 32 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'LRCN' -n '917/labels_1' -i 128 --label 1 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'LRCN' -n '917/labels_8' -i 128 --label 8 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'LRCN' -n '917/labels_16' -i 128 --label 16 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'LRCN' -n '917/labels_32' -i 128 --label 32 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'video' -n '917/labels_1' -i 128 --label 1 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'video' -n '917/labels_8' -i 128 --label 8 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'video' -n '917/labels_16' -i 128 --label 16 --img_width 128 --VM 917
python3 ConvLSTM_train.py -m 'video' -n '917/labels_32' -i 128 --label 32 --img_width 128 --VM 917




