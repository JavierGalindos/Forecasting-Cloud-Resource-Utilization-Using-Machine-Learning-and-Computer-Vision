import os
import matplotlib.pyplot as plt
import pandas as pd

# Constants
LOGS_PATH = './logs'
HP = 'neurons'
HP_NAME = 'neurons'
FIGURES_PATH = '../Figures/Modeling/LSTM'
if __name__ == "__main__":
    # List files
    hp_list =[]
    for file in os.listdir(LOGS_PATH):
        if file.startswith(HP):
            hp_list.append(file)

    # List files with specific hyper-parameters
    hp_df_list = []
    for idx, file in enumerate(hp_list):
        data = pd.read_csv(os.path.join(LOGS_PATH, hp_list[idx], 'metrics.txt'),
                           names=[int(hp_list[idx].split('_')[1])], sep='   ', engine='python')
        hp_df_list.append(data)

    # Create dataframe and sort
    hp_df = pd.concat(hp_df_list, axis=1)
    hp_df.sort_index(axis=1, inplace=True)

    # Generate figures and save them
    # MAE & MAE
    fig = plt.figure(dpi=200)
    hp_df.iloc[0:2, :].plot.bar(rot=0)
    plt.xlabel('Metric')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.mkdir(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'MAE_MAPE')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    # MSE
    fig = plt.figure(dpi=200)
    hp_df.iloc[2, :].plot.bar(rot=0)
    plt.xlabel(f'{HP_NAME}')
    plt.ylabel('MSE')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.mkdir(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'MSE')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


