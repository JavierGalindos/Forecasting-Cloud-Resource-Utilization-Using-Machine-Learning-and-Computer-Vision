import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import random
from typing import List
import seaborn as sns

''' Data constants'''
DATA_PATH = r'../Datasets/fastStorage/2013-8'
FIGURES_PATH = '../Figures/DataExploration/'


# Uncomment when using Data exploration
# if not os.access(FIGURES_PATH, os.F_OK):
#     os.mkdir(FIGURES_PATH)
# if not os.access(FIGURES_PATH, os.W_OK):
#     print('Cannot write to {}, please fix it.'.format(FIGURES_PATH))
#     exit()
# else:
#     print('figures saved to {}'.format(FIGURES_PATH))


def load_VM(VM_name: str) -> pd.DataFrame:
    """ Creates a dataframe for each VM

    Parameters
    ----------
    VM_name
        name of the VM. (e.g., '1.csv')

    Returns
    -------
    Pandas DataFrame of VMs
    """
    VM_path = os.path.join(DATA_PATH, VM_name)
    # Read time series of each Virtual Machine (VM)
    VM = pd.read_csv(VM_path, sep=';\t', engine='python')
    # Parse dates
    VM['Timestamp [ms]'] = pd.to_datetime(VM['Timestamp [ms]'], unit='s')
    VM = VM.set_index('Timestamp [ms]')
    # Create new variable memory usage in %
    VM['Memory usage [%]'] = VM['Memory usage [KB]'] * 100 / VM['Memory capacity provisioned [KB]']
    # Avoid division by 0
    VM['Memory usage [%]'] = VM['Memory usage [%]'].fillna(0)
    # Group by index and average (avoid duplicate timestamps)
    VM = VM.groupby(VM.index).mean()
    # Floor time index
    VM.index = VM.index.floor(freq='5T')
    VM = VM.groupby(VM.index).mean()
    # Avoid NaN and Inf values
    VM.replace([np.inf, -np.inf], np.nan, inplace=True)
    VM.dropna(inplace=True)
    return VM


def load_all_VMs(data_path: str = DATA_PATH) -> List[pd.DataFrame]:
    """ Creates a list of DataFrames of VMs

    Parameters
    ----------
    data_path

    Returns
    -------
    list of DataFrames of each VM
    """
    files = os.listdir(data_path)  # Get all the files in that directory
    files.sort()  # Short the files (compatible with mac)
    datacenter = []
    for idx, serie in enumerate(files):
        VM = load_VM(serie)
        datacenter.append(VM)
    return datacenter


def filter_VMs(VMs: List[pd.DataFrame], mean_th: int = 100, std_th: int = 50, max_th: int = 1000) -> List[pd.DataFrame]:
    """ Filter the VMs that does not give useful information and distort the algorithms

    Parameters
    ----------
    VMs
        list of VMs
    mean_th
        mean threshold
    std_th
        std threshold
    max_th
        max thresholds

    Returns
    -------
    VMs_training
        List of useful VMs
    """
    VMs_training = [VM for VM in VMs if (VM['CPU usage [MHZ]'].mean() > mean_th) or
                    (VM['CPU usage [MHZ]'].std() > std_th) or
                    (VM['CPU usage [MHZ]'].max() > max_th)]
    return VMs_training


def load_datacenter(VMs: List[pd.DataFrame]) -> pd.DataFrame:
    """ Creates a dataframe of the whole datacenter (sum of VMs)

    Parameters
    ----------
    VMs
        list of VMs

    Returns
    -------
    Pandas DataFrame of the datacenter
    """
    # Merge all the VMs and average every variable
    datacenter = pd.concat(VMs).groupby('Timestamp [ms]').mean()
    # Uncomment the following lines whether you want to accumulate variables but %
    # # Sum everything but [%] variables (mean)
    # datacenter = pd.concat(
    #     [datacenter, VM.loc[:, ~VM.columns.isin(['CPU usage [%]', 'Memory usage [%]'])]]).groupby(
    #     'Timestamp [ms]').sum()
    # # Mean in [%] variables
    # datacenter = pd.concat([datacenter, VM[['CPU usage [%]', 'Memory usage [%]']]]).groupby('Timestamp [ms]').mean()
    return datacenter


def plot_timeSeries(data, MA=0, ema=0.05, legend=True, xlabel='Time', ylabel=None, title=None, xlim=None, ylim=None,
                    xticks=None, figsize=(8, 6), dpi=120, savefig=None, show=True, **kwargs):
    """ Utility to plot Time Series with moving average filters """

    # Define default kwargs
    defaultKwargs = {'marker': 'o',
                     'linestyle': '',
                     'alpha': 0.3,
                     'markersize': 2}
    kwargs = {**defaultKwargs, **kwargs}
    if show is not False:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    if type(data) == pd.DataFrame:
        data.dropna(inplace=True)
    data.plot(label='Raw data', **kwargs)
    if MA > 0: data.rolling(window=MA).mean().plot(label='SMA {}'.format(MA), alpha=0.5)
    if ema > 0: data.ewm(alpha=ema, adjust=False).mean().plot(label='EMA {}'.format(ema), alpha=0.5)
    if legend: plt.legend()
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if title is not None: plt.title(title)
    if xticks is not None: plt.xticks(xticks)
    if savefig is not None:
        save_path = os.path.join(FIGURES_PATH, savefig)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        if show is True:
            plt.show()


# Descriptive statistics

def descriptive_stats(VMs: List[pd.DataFrame], features: List[str] = 'CPU usage [MHZ]') -> pd.DataFrame:
    """ Create a DataFrame of descriptive statistics of the VMs

    Parameters
    ----------
    VMs
        List of VM DataFrames
    features
        features to compute the statistics
    Returns
    -------
    descriptive_stats_df
        DataFrame of descriptive statistics of the VMs
    """

    stats = []
    cum_sum = []
    for idx, VM in enumerate(VMs):
        # Mean, min, max, median, quantiles, etc.
        stats.append(VM[features].describe())
        # Sum over the series
        cum_sum.append(VM[features].sum())
    # Create dataframe
    descriptive_stats_df = pd.DataFrame(stats, index=range(len(VMs)))
    descriptive_stats_df['sum'] = cum_sum
    return descriptive_stats_df


def plot_stats(stats_df: pd.DataFrame, features: List[str] = 'CPU usage [MHZ]', savefig=None):
    """ Plot descriptive statistics

    Parameters
    ----------
    stats_df
    features
    savefig
    """
    # Create the folder whether not exists
    if savefig is not None:
        if not os.path.exists(os.path.join(FIGURES_PATH, savefig)):
            os.makedirs(os.path.join(FIGURES_PATH, savefig))
    # Bar plots
    for column in stats_df.columns:
        fig = plt.figure()
        plt.title('Descriptive statistics {}: {}'.format(features, column))
        plt.bar(np.arange(1250), stats_df[column])
        plt.xlabel('VM')
        plt.ylabel(features)
        if savefig is not None:
            save_path = os.path.join(FIGURES_PATH, savefig, 'barplot_{}'.format(column))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    # Histograms
    for column in stats_df.columns:
        fig = plt.figure()
        sns.histplot(data=stats_df, x=column).set(title='Histogram of {} {}'.format(column, features))
        if savefig is not None:
            save_path = os.path.join(FIGURES_PATH, savefig, 'histogram_{}'.format(column))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


# Metrics
def mase(y, y_hat, y_train) -> np.ndarray:
    """ Mean Absolute Scale Error.
    Parameters
    ----------
    y
    y_hat
    y_train

    Returns
    -------

    """
    # Naive in-sample Forecast
    naive_y_hat = y_train[:-1]
    naive_y = y_train[1:]

    # Calculate MAE (in sample)
    mae_in_sample = np.mean(np.abs(naive_y - naive_y_hat))

    mae = np.mean(np.abs(y - y_hat))

    return mae / mae_in_sample
