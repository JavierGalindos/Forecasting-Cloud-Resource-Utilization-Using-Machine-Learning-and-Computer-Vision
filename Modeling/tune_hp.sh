#! /bin/bash
python3 ConvLSTM_CM.py -m 'AE' -n '917/CM_251' -i 128 --VM 917 --label 16 --VM_test 251
python3 ConvLSTM_CM.py -m 'LRCN' -n '917/CM_251' -i 128 --VM 917 --label 16 --VM_test 251
python3 ConvLSTM_CM.py -m 'video' -n '917/CM_251' -i 128 --VM 917 --label 16 --VM_test 251
python3 ConvLSTM_CM.py -m 'AE' -n '917/CM_322' -i 128 --VM 917 --label 16 --VM_test 322
python3 ConvLSTM_CM.py -m 'LRCN' -n '917/CM_322' -i 128 --VM 917 --label 16 --VM_test 322
python3 ConvLSTM_CM.py -m 'video' -n '917/CM_322' -i 128 --VM 917 --label 16 --VM_test 322





