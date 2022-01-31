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

if not os.access(FIGURES_PATH, os.F_OK):
    os.mkdir(FIGURES_PATH)
if not os.access(FIGURES_PATH, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(FIGURES_PATH))
    exit()
else:
    print('figures saved to {}'.format(FIGURES_PATH))


def load_VM(VM_name: str) -> pd.DataFrame:
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
        # Average every variable
        datacenter = pd.concat([datacenter, VM]).groupby('Timestamp [ms]').mean()
        # Uncomment the following lines whether you want to accumulate variables but %
        # # Sum everything but [%] variables (mean)
        # datacenter = pd.concat(
        #     [datacenter, VM.loc[:, ~VM.columns.isin(['CPU usage [%]', 'Memory usage [%]'])]]).groupby(
        #     'Timestamp [ms]').sum()
        # # Mean in [%] variables
        # datacenter = pd.concat([datacenter, VM[['CPU usage [%]', 'Memory usage [%]']]]).groupby('Timestamp [ms]').mean()
    return datacenter


def clustering_preprocessing(VMs: List[pd.DataFrame], features: List[str], length: int):
    """ Preprocessing for clustering
        Select only one feature & shorten the series due to time limitation
    :param VMs: list of VMs
    :param length: length of shorten time series
    """
    # Feature selection:
    VMs_fs = [VM[features] for VM in VMs]
    VMs_fs_ts = to_time_series_dataset(VMs_fs)
    # Shorten to x timestamps
    VMs_fs_short = TimeSeriesResampler(sz=length).fit_transform(VMs_fs)
    VMs_fs_short_ts = to_time_series_dataset(VMs_fs_short)
    return VMs_fs_ts, VMs_fs_short_ts


def load_clusters(VMs: List[pd.DataFrame], labels: np.ndarray, n_clusters: int) -> List[pd.DataFrame]:
    """ Creates a dataframe per cluster as an addition of VMs within this cluster

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
            # Average on every variable
            cluster = pd.concat([cluster, VM]).groupby('Timestamp [ms]').mean()
            # # Sum everything but [%] variables (mean)
            # cluster = pd.concat(
            #     [cluster, VM.loc[:, ~VM.columns.isin(['CPU usage [%]', 'Memory usage [%]'])]]).groupby(
            #     'Timestamp [ms]').sum()
            # # Mean in [%] variables
            # cluster = pd.concat([cluster, VM[['CPU usage [%]', 'Memory usage [%]']]]).groupby(
            #     'Timestamp [ms]').mean()
        # Append cluster
        clusters.append(cluster)
    return clusters


def plot_timeSeries(data, MA=10, ema=0.05, legend=True, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None,
                    xticks=None, figsize=(8, 6), dpi=120, savefig=None, show=True):
    """ Utility to plot Time Series with moving average filters """

    if show is not False:
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
        if show is True:
            plt.show()


def plot_clusters(data, labels, clusters, n_clusters, shared_axis=False, legend=False, xlabel='Timestmaps [ms]',
                  ylabel=None, title=None, xlim=None, ylim=None, xticks=None, figsize=(13, 4), dpi=120, savefig=None):
    """ Utility to plot VMs of each cluster

        Generate two subfigures:
            Left: Overlapped VMs
            Right: Average features per cluster of VMs
    """
    # TODO: check implentation of shared axis
    if n_clusters is None: n_clusters = len(np.unique(labels))
    for cluster_num in range(n_clusters):
        # List of VMs of this cluster
        VMs_cluster = [VM for idx, VM in enumerate(data) if labels[idx] == cluster_num]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        title_fig = title + ' Cluster {} ({} VMs)'.format(cluster_num, len(VMs_cluster))
        if title is not None: fig.suptitle(title_fig, fontsize=14)
        # Right figure
        if clusters is not None:
            ax1 = plt.subplot(1, 2, 2)
            cluster = clusters[cluster_num]
            plot_timeSeries(cluster.iloc[:, 0],
                            title='Average over the cluster',
                            legend=True,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            xlim=xlim,
                            ylim=ylim,
                            xticks=xticks,
                            figsize=(4, 4),
                            dpi=dpi,
                            savefig=None,
                            show=False)
        # Left figure
        if clusters is not None:
            if shared_axis:
                plt.subplot(1, 2, 1, sharey=ax1, visible=True)
            else:
                plt.subplot(1, 2, 1)
        for VM in VMs_cluster:
            plt.plot(VM, "k-", alpha=.2)
        if legend: plt.legend()
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if xticks is not None: plt.xticks(xticks)
        plt.title('VMs in the cluster')

        if savefig is not None:
            save_path = os.path.join(FIGURES_PATH, savefig, 'Cluster{}'.format(cluster_num))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def optimal_clusters(features: List[str], models_path: str, length: int = 500, fast_validation: bool = True):
    # Load all VMs (list of VMs)
    VMs = load_all_VMs()
    print('Load VM: Completed')
    # Select only one feature & shorten the series due to time limitation
    VMs_fs_ts, VMs_fs_short_ts = clustering_preprocessing(VMs, features=features, length=length)
    print('Preprocessing: Completed')
    # Loop through different configurations for # of clusters and store the respective values for silhouette:
    sil_scores = []
    for n_clusters in range(2, 10):
        print('k-means ({} clusters):'.format(n_clusters))
        # Set model options
        kmeans_model = TimeSeriesKMeans(n_clusters=n_clusters,
                                        metric="dtw",
                                        n_jobs=-1,  # All available workers
                                        verbose=True)
        # Train the model
        kmeans_model.fit(VMs_fs_short_ts)
        # Save the model to disk
        if not os.access(models_path, os.F_OK):
            os.mkdir(models_path)
        if not os.access(models_path, os.W_OK):
            print('Cannot write to {}, please fix it.'.format(models_path))
            exit()
        save_path = os.path.join(models_path, 'kmeans_{}.hdf5'.format(n_clusters))
        kmeans_model.to_hdf5(save_path)
        # Predict labels
        labels = kmeans_model.predict(VMs_fs_short_ts)
        # Silhouette coefficient
        if fast_validation is True:
            sil_scores.append(silhouette_score(VMs_fs_short_ts,
                                               labels,
                                               metric="dtw",
                                               n_jobs=-1,
                                               verbose=True))
        else:
            sil_scores.append(silhouette_score(VMs_fs_ts,
                                               labels,
                                               metric="dtw",
                                               n_jobs=-1,
                                               verbose=True))
    return sil_scores


def plot_silhouette(sil_score: List[float], legend=True, xlabel='# of clusters', ylabel='Silhouette coefficient',
                    title='K-means clustering', xlim=None, ylim=None, xticks=np.arange(2, 10), figsize=(8, 6),
                    dpi=120, savefig=None):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(np.arange(2, 10), sil_score)
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
