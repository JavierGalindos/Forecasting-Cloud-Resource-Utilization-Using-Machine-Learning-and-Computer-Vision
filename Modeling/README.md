# Modeling

## LSTM

## Load Tensorboard
To check training process:
```
tensorboard --logdir './Modeling/darts_logs/{model_name}/logs'  
```
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

