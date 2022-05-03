import os
import matplotlib.pyplot as plt
import pandas as pd

# Constants
LOGS_PATH = './logs/Comparison/CM322'
HP = 'CM322'
HP_NAME = 'Inference VM 322'
FIGURES_PATH = '../Figures/Modeling/Comparison/CM322'
plt.rcParams['font.size'] = '13'
if __name__ == "__main__":
    # List files
    hp_list =[]
    for file in os.listdir(LOGS_PATH):
        if file.startswith(HP):
            hp_list.append(file)

    # List files with specific hyper-parameters
    hp_df_list = []
    for idx, file in enumerate(hp_list):
        # Remove int() if parameters are not numeric
        # data = pd.read_csv(os.path.join(LOGS_PATH, hp_list[idx], 'metrics.txt'),
        #                    names=[int(hp_list[idx].split('_')[1])], sep=',', index_col=0, engine='python')
        data = pd.read_csv(os.path.join(LOGS_PATH, hp_list[idx], 'metrics.txt'),
                            names=[hp_list[idx].split('_')[1]], sep=',', index_col=0, engine='python')
        hp_df_list.append(data)

    # Create dataframe and sort
    hp_df = pd.concat(hp_df_list, axis=1)
    hp_df.sort_index(axis=1, inplace=True)
    hp_df = hp_df.T
    hp_df['model_size [B]'] = hp_df['model_size [B]'] * 0.000001
    hp_df = hp_df.rename(columns={'model_size [B]': 'model_size [MB]'})
    hp_df = hp_df.T
    print(hp_df.round(3).T)

    # Generate figures and save them
    # MAE, MAPE & RMSE
    fig = plt.figure(dpi=200, figsize=(5, 3))
    hp_df.iloc[1:4, :].plot.bar(rot=0)
    plt.xlabel('Metric')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.makedirs(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'MAE_MAPE_RMSE')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Training time
    fig = plt.figure(dpi=200, figsize=(5, 3))
    hp_df.iloc[5, :].plot.bar(rot=0, width=0.2)
    plt.xlabel(f'{HP_NAME}')
    plt.ylabel('Training time [s]')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.mkdir(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'train_time')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Inference time
    fig = plt.figure(dpi=200, figsize=(5, 3))
    hp_df.iloc[6, :].plot.bar(rot=0, width=0.2, color='tab:orange')
    plt.xlabel(f'{HP_NAME}')
    plt.ylabel('Inference time [s]')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.mkdir(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'infer_time')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Size of the mode
    fig = plt.figure(dpi=200, figsize=(5, 3))
    hp_df.iloc[7, :].plot.bar(rot=0, width=0.2, color='tab:green')
    plt.xlabel(f'{HP_NAME}')
    plt.ylabel('Size of the model [B]')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.mkdir(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'model_size')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # IoU
    fig = plt.figure(dpi=200, figsize=(5, 3))
    hp_df.iloc[8, :].plot.bar(rot=0, width=0.2, color='tab:purple')
    plt.xlabel(f'{HP_NAME}')
    plt.ylabel('IoU')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.mkdir(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'IoU')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # DTW
    fig = plt.figure(dpi=200, figsize=(5, 3))
    hp_df.iloc[9, :].plot.bar(rot=0, width=0.2, color='tab:red')
    plt.xlabel(f'{HP_NAME}')
    plt.ylabel('Dynamic time warping')
    plt.title(f'Hyperparameter: {HP_NAME}')
    if not os.access(os.path.join(FIGURES_PATH, HP_NAME), os.F_OK):
        os.mkdir(os.path.join(FIGURES_PATH, HP_NAME))
    save_path = os.path.join(FIGURES_PATH, HP_NAME, 'DTW')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


