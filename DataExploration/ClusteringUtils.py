from BitbrainsUtils import *
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesResampler
from typing import List


def clustering_preprocessing(VMs: List[pd.DataFrame], features: List[str], length: int):
    """Preprocessing for clustering
      Select only one feature & shorten the series due to time limitation

    Parameters
    ----------
    VMs
        list of VMs
    features
    length
        length of shorten time series

    Returns
    -------

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

    Parameters
    ----------
    VMs
        list of VMs Dataframes
    labels
        cluster assigned for each VM
    n_clusters

    Returns
    -------
    list of clusters DataFrames
    """
    clusters = []
    if n_clusters != len(np.unique(labels)):
        assert "Number of clusters is not correct"
    for cluster_num in range(n_clusters):
        # List of VMs of this cluster
        VMs_cluster = [VM for idx, VM in enumerate(VMs) if labels[idx] == cluster_num]
        # Average on every variable
        cluster = pd.concat(VMs_cluster).groupby('Timestamp [ms]').mean()
        # Append cluster
        clusters.append(cluster)
    return clusters


def plot_clusters(data, labels, clusters, n_clusters, shared_axis=False, filters=True, marker='k-', legend=False,
                  xlabel='Time', ylabel=None, title=None, xlim=None, ylim=None, xticks=None,
                  figsize=(13, 6), dpi=120, savefig=None, **kwargs):
    """ Utility to plot VMs of each cluster

        Generate two subfigures:
            Left: Overlapped VMs
            Right: Average features per cluster of VMs
    """
    if n_clusters is None: n_clusters = len(np.unique(labels))
    for cluster_num in range(n_clusters):
        # List of VMs of this cluster
        VMs_cluster = [VM for idx, VM in enumerate(data) if labels[idx] == cluster_num]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.tight_layout()
        title_fig = title + ' Cluster {} ({} VMs)'.format(cluster_num, len(VMs_cluster))
        if title is not None: fig.suptitle(title_fig, fontsize=14)
        # Right figure
        if clusters is not None:
            ax1 = plt.subplot(1, 2, 2)
            cluster = clusters[cluster_num]
            if filters:
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
                                show=False,
                                **kwargs)
            else:
                plot_timeSeries(cluster.iloc[:, 0],
                                MA=0,
                                ema=0,
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
                                show=False,
                                **kwargs)
        # Left figure
        if clusters is not None:
            if shared_axis:
                plt.subplot(1, 2, 1, sharey=ax1, visible=True)
            else:
                plt.subplot(1, 2, 1)
        for VM in VMs_cluster:
            plt.plot(VM, marker, alpha=.2, markersize=0.1)
            plt.xticks(rotation=30)
        if legend: plt.legend()
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if xticks is not None: plt.xticks(xticks, rotation=45)
        plt.title('VMs in the cluster')

        if savefig is not None:
            # Create the folder whether not exists
            if not os.path.exists(os.path.join(FIGURES_PATH, savefig)):
                os.makedirs(os.path.join(FIGURES_PATH, savefig))
            save_path = os.path.join(FIGURES_PATH, savefig, 'Cluster{}'.format(cluster_num))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def optimal_clusters(data, features: List[str], models_path: str, length: int = 500, fast_validation: bool = True):
    """ Returns the silhouette score for different k using k-means towards finding optimal number of clusters

    Parameters
    ----------
    data
    features
    models_path
    length
    fast_validation

    Returns
    -------

    """
    if data is None:
        # Load all VMs (list of VMs)
        data = load_all_VMs()
        print('Load VM: Completed')
    # Select only one feature & shorten the series due to time limitation
    VMs_fs_ts, VMs_fs_short_ts = clustering_preprocessing(data, features=features, length=length)
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
        try:
            kmeans_model.to_hdf5(save_path)
        except FileExistsError:
            print('Model already saved - Overwrite')
            # Remove previous file and save it again
            os.remove(save_path)
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


def plot_silhouette(sil_score: List[float], legend=False, xlabel='# of clusters', ylabel='Silhouette coefficient',
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


def get_silhouette_from_models(models_path: str, VMs_ts: np.ndarray) -> List[float]:
    """ Evaluate the already trained models and get silhouette scores

    Parameters
    ----------
    models_path: str
        path to the models
    VMs_ts:
        Time series of the VMs (tslearn)

    Returns
    -------
    sil_score
    """
    # Get all the models
    files = os.listdir(models_path)  # Get all the files in that directory
    # Models ends with .hdf5
    models = [_ for _ in files if _.endswith('.hdf5')]
    sil_score = []
    for _, model in enumerate(models):
        # Load model
        try:
            kmeans_model = TimeSeriesKMeans.from_hdf5(os.path.join(models_path, model))
        except OSError:
            print("Could not open the file: {}".format(model))
        # Predict labels
        labels = kmeans_model.predict(VMs_ts)
        sil_score.append(silhouette_score(VMs_ts,
                                          labels,
                                          metric="dtw",
                                          n_jobs=-1,
                                          verbose=True))
    return sil_score


def elbow_method(models_path: str) -> List[float]:
    """ Return the intertia for the elbow method
        The k-means models have to have been previously created

    Parameters
    ----------
    models_path
        path to the models

    Returns
    -------
    list of clusters DataFrames
    """
    # Get all the models
    files = os.listdir(models_path)  # Get all the files in that directory
    # Models ends with .hdf5
    models = [_ for _ in files if _.endswith('.hdf5')]
    inertia = []
    for _, model in enumerate(models):
        # Load model
        try:
            kmeans_model = TimeSeriesKMeans.from_hdf5(os.path.join(models_path, model))
        except OSError:
            print("Could not open the file: {}".format(model))
        inertia.append(kmeans_model.inertia_)
    return inertia


def split_cluster(VMs: List[pd.DataFrame], labels: np.ndarray, model_big_cluster: str, n_clusters: int,
                  big_cluster_num: int, features: List[str]) -> np.ndarray:
    """ Split the big cluster in sub-clusters
        Returns the new labels with the sub-clusters

    Parameters
    ----------
    VMs
        List of VM DataFrames
    labels
        cluster assigned for each VM
    model_big_cluster
        directory of the k-means model. Ex: './kmeans_models/big_cluster/kmeans_2.hdf5'
    n_clusters
    big_cluster_num
        number of the big cluster
    features

    Returns
    -------
    labels with sub-clusters
    """
    # Copy labels
    new_labels = np.copy(labels)

    big_cluster = [VM for idx, VM in enumerate(VMs) if labels[idx] == big_cluster_num]
    VMs_fs_ts, VMs_fs_short_ts = clustering_preprocessing(big_cluster, features=features, length=500)
    # Load model
    try:
        kmeans_model = TimeSeriesKMeans.from_hdf5(model_big_cluster)
    except OSError:
        print("Could not open the file: {}".format(model_big_cluster))
    # cluster assigned for each VM within the big cluster
    labels_bigCluster = kmeans_model.predict(VMs_fs_short_ts)
    idx_2 = 0  # Index for big cluster
    # Loop over every VM
    for idx, VM in enumerate(VMs):
        # If the VM is in the big cluster, change the label
        if labels[idx] == big_cluster_num:
            new_labels[idx] = labels_bigCluster[idx_2] + n_clusters
            idx_2 += 1  # Go to next element of big cluster
    new_labels[new_labels == max(new_labels)] = 0  # First cluster number is 0 for correct indexing
    return new_labels
