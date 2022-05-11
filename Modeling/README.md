# Modeling

# Baseline

Baseline.py contains the class Baseline to run traditional time series forecasting models (ARIMA & Exponential Smoothing)

To run:
```
python Baseline_train.py -m <model_name> -n <name of the model> --label <labels length> --VM <Virtual Machine number> --VM <Virtual Machine number for inference>
```

## LSTM
LSTM.py contains the class LstmModel to run the RNN models.

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

## ConvLSTM
ConvLSTM.py contains the class ConvLSTMModel to run the image-based models, present in model_zoo.py:
- frame: Model used in Keras tutorial for next video-frame prediction
- LRCN: Long Recurrent Convolution Neural Network (CNN + RNN)
- ConvLSTM: ConvLSTM with down sampling
- video: Video frame prediction model
- AE: Convolutional Autoencoder


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

### Inference in a new test set: ConvLSTM_CM.py
Inference in a different VM that the one the model was trained on.

To run:
```
python ConvLSTM_CM.py -e <number of epoch> -m <model_name> -n <name of the model>  -i <input_length>   --label <labels length> -f <number of frames> 
```







## Utils
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
### Run jypyter in ssh server
Remote terminal:
```
export PYTHONPATH='/home/javier.galindos/intern-project-javier/'
jupyter notebook --no-browser --port=8080
```

Local terminal:
```
ssh -N -L 8080:localhost:8080 javier.galindos@machine-learning.imdea
```


### Python path
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

### Tune hyper-parameters
Modify the shell script [tune_hp.sh](tune_hp.sh) to tune desired hyperparameters
```
chmod +x tune_hp.sh
./tune_hp.sh
```