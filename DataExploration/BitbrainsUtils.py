import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesResampler
from typing import List

''' Data constants'''
DATA_PATH = r'../Datasets/fastStorage/2013-8'
FIGURES_PATH = '../Figures/DataExploration/'


def load_VM(VM_name: str) -> object:
    """ Creates a dataframe for each VM

    :param VM_name: name of the VM. (e.g., '1.csv')
    :return: Pandas DataFrame of VMs
    """
    VM_path = os.path.join(DATA_PATH, VM_name)
    # Read time series of each Virtual Machine (VM)
    VM = pd.read_csv(VM_path, sep=';\t', index_col=0, parse_dates=True, squeeze=True, engine='python')
    # Create new variable memory usage in %
    VM['Memory usage [%]'] = VM['Memory usage [KB]'] * 100 / VM['Memory capacity provisioned [KB]']
    # Avoid division by 0
    VM['Memory usage [%]'] = VM['Memory usage [%]'].fillna(0)
    # Group by index and average (avoid duplicate timestamps)
    VM = VM.groupby(VM.index).mean()
    # Avoid NaN and Inf values
    VM.replace([np.inf, -np.inf], np.nan, inplace=True)
    VM.dropna(inplace=True)
    return VM


def load_all_VMs(data_path: str = DATA_PATH) -> List[pd.DataFrame]:
    """ Creates a list of DataFrames of VMs

    :param data_path:
    :return: list of DataFrames of each VM
    """
    files = os.listdir(data_path)  # Get all the files in that directory
    datacenter = []
    for idx, serie in enumerate(files):
        VM = load_VM(serie)
        datacenter.append(VM)
    return datacenter


def load_datacenter(data_path: str = DATA_PATH) -> pd.DataFrame:
    """ Creates a dataframe of the whole datacenter (sum of VMs)

    :param data_path:
    :return: Pandas DataFrame of the datacenter
    """
    # New dataframe
    datacenter = pd.DataFrame()
    # Get all the series
    files = os.listdir(data_path)  # Get all the files in that directory
    for serie in files:
        VM = load_VM(serie)
        # Sum everything but [%] variables (mean)
        datacenter = pd.concat(
            [datacenter, VM.loc[:, ~VM.columns.isin(['CPU usage [%]', 'Memory usage [%]'])]]).groupby(
            'Timestamp [ms]').sum()
        # Mean in [%] variables
        datacenter = pd.concat([datacenter, VM[['CPU usage [%]', 'Memory usage [%]']]]).groupby('Timestamp [ms]').mean()
    return datacenter


def clustering_preprocessing(VMs: List[pd.DataFrame], length: int):
    """ Preprocessing for clustering
        Select only one feature & shorten the series due to time limitation
    :param VMs: list of VMs
    :param length: length of shorten time series
    """
    # Feature selection: CPU usage [MHZ]
    VMs_CPU = [VM[['CPU usage [MHZ]']] for VM in VMs]
    VMs_CPU_ts = to_time_series_dataset(VMs_CPU)
    # Shorten to x timestamps
    VMs_CPU_short = TimeSeriesResampler(sz=length).fit_transform(VMs_CPU)
    VMs_CPU_short_ts = to_time_series_dataset(VMs_CPU_short)
    return VMs_CPU_ts, VMs_CPU_short_ts


def load_clusters(VMs: List[pd.DataFrame], labels: np.ndarray, n_clusters: int) -> List[pd.DataFrame]:
    """ Creates a dataframe per cluster as an addition of VMs whithin this cluster

    :param VMs: list of VMs Dataframes
    :param labels: cluster assigned for each VM
    :param n_clusters:
    :return: list of clusters DataFrames
    """
    clusters = []
    if n_clusters != len(np.unique(labels)):
        assert "Number of clusters is not correct"
    for cluster_num in range(n_clusters):
        # Loop over VMs within a cluster
        cluster = pd.DataFrame()
        # List of VMs of this cluster
        VMs_cluster = [VM for idx, VM in enumerate(VMs) if labels[idx] == cluster_num]
        for VM in VMs_cluster:
            # Sum everything but [%] variables (mean)
            cluster = pd.concat(
                [cluster, VM.loc[:, ~VM.columns.isin(['CPU usage [%]', 'Memory usage [%]'])]]).groupby(
                'Timestamp [ms]').sum()
            # Mean in [%] variables
            cluster = pd.concat([cluster, VM[['CPU usage [%]', 'Memory usage [%]']]]).groupby(
                'Timestamp [ms]').mean()
        # Append cluster
        clusters.append(cluster)
    return clusters


def plot_timeSeries(data, MA=10, ema=0.05, legend=True, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None,
                    xticks=None, figsize=(8, 6), dpi=120, savefig=None):
    """ Utility to plot Time Series with moving average filters """

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if type(data) == pd.DataFrame:
        data.dropna(inplace=True)

    data.plot(label='Raw data')
    if MA > 0: data.rolling(window=MA).mean().plot(label='MA({}) filter'.format(MA))
    if ema > 0: data.ewm(alpha=ema, adjust=False).mean().plot(label='EMA {}'.format(ema))
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
        plt.show()


def plot_clusters(data, labels, n_clusters, legend=False, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None,
                  xticks=None, figsize=(8, 6), dpi=120, savefig=None):
    """ Utility to plot VMs of each cluster """

    for yi in range(n_clusters):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        for xx in data[labels == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        if legend: plt.legend()
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if title is not None: plt.title(title)
        if xticks is not None: plt.xticks(xticks)
        if savefig is not None:
            save_path = os.path.join(FIGURES_PATH, savefig, 'Cluster{}'.format(yi))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
