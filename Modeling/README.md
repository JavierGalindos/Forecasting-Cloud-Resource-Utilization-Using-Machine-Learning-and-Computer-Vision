# Modeling

## LSTM


### LSTM_train.py
Trains an LSTM model using Keras and Tensorflow.

To run:
```
python LSTM_train.py -e <number of epoch> -n <name of the model> -s <path to save the figure> -i <input_length> -d <number of neurons> -l <number or rnn layers --dropout <droput rate>  --label <labels length>
```

Example:
```
export PYTHONPATH='/Users/javier.galindos/Library/CloudStorage/OneDrive-FundaciónIMDEASoftware/Documents/Code/intern-project-javier'
python3 LSTM_train.py -e 100 -n 'input_50' -i 50 -s 'input_length'
```

### Load Tensorboard
To check training process:
```
tensorboard --logdir='Modeling/logs/{model_name}/tensorboard' 
```

If running from remote server (port forwarding)
Local terminal:
```
ssh -N -L 6006:localhost:6006 javier.galindos@machine-learning.imdea
```

### Tune hyper-parameters
Modify the shell script [tune_hp.sh](tune_hp.sh) to tune desired hyperparameters
```
chmod +x tune_hp.sh
./tune_hp.sh
```

## ConvLSTM

### ConvLSTM_train.py
Trains an ConvLSTM model using Keras and Tensorflow.

To run:
```
python ConvLSTM_train.py -e <number of epoch> -m <model_name> -n <name of the model>  -i <input_length>   --label <labels length> -f <number of frames> 
```

Example:
```
export PYTHONPATH='/Users/javier.galindos/Library/CloudStorage/OneDrive-FundaciónIMDEASoftware/Documents/Code/intern-project-javier'
python3 ConvLSTM_train.py -e 100 -n '917/ConvLSTM' -i 64 --label 16
```

### Load Tensorboard
To check training process:
```
tensorboard --logdir='Modeling/logs/ConvLSTM/{model_name}/tensorboard' 
```



## Run jypyter in ssh server
Remote terminal:
```
export PYTHONPATH='/home/javier.galindos/intern-project-javier/'
jupyter notebook --no-browser --port=8080
```

Local terminal:
```
ssh -N -L 8080:localhost:8080 javier.galindos@machine-learning.imdea
```

## Darts
### train_LSTM.py
Trains an LSTM model over the Darts framework.

To run:
```
python train_LSTM.py -e <number of epoch> -n <name of the model> -s <path to save the figure> -i <input_chunck_length> -d <number of neurons> -l <number or rnn layers --dropout <droput rate> --lr <learning rate> --fh <forecasting horizon>
```

Example:
```
export PYTHONPATH='/Users/javier.galindos/Library/CloudStorage/OneDrive-FundaciónIMDEASoftware/Documents/Code/intern-project-javier'
python3 train_LSTM.py -e 20 -n '_input_1day' -i 288 -s 'LSTM/input_chunk'
```

# Python path
Local:
```
export PYTHONPATH='/Users/javier.galindos/Library/CloudStorage/OneDrive-FundaciónIMDEASoftware/Documents/Code/intern-project-javier'
```
VM:
```
export PYTHONPATH='/home/javier.galindos/intern-project-javier/'
```
GPU:
```
export PYTHONPATH=~/intern-project-javier/
```