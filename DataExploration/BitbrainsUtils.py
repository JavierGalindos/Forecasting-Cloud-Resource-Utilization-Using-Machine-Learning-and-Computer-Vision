import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import random
from typing import List
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

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
                    xticks=None, figsize=(15, 2), dpi=120, savefig=None, show=True, **kwargs):
    """ Utility to plot Time Series with moving average filters """

    # Define default kwargs
    defaultKwargs = {'marker': 'o',
                     'linestyle': '-',
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
def mae_array(y, y_hat):
    """ Mean Absolute Error.
        Return the array of the individual errors
    Parameters
    ----------
    y
        ground truth
    y_hat
        prediction

    Returns
    -------

    """
    return np.abs(y - y_hat)


def rmse_array(y, y_hat):
    """ Root Mean Squared Error
        Return the array of the individual errors
    Parameters
    ----------
    y
        ground truth
    y_hat
        prediction

    Returns
    -------

    """
    return np.sqrt(np.square(y - y_hat))


def mape_array(y, y_hat):
    """ Mean Absolute Percentage Error
            Return the array of the individual errors
        Parameters
        ----------
        y
            ground truth
        y_hat
            prediction

        Returns
        -------

        """
    return np.abs((y - y_hat) / y) * 100


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


def split_data(df: pd.DataFrame, training: float = 0.7, validation: float = 0.2, test: float = 0.1):
    """ Split the dataset in train, validation and test sets

    Parameters
    ----------
    df
    training
    validation
    test

    Returns
    -------

    """
    n = len(df)
    df_copy = df.copy()
    train_df = df_copy[0:int(n * training)]
    val_df = df_copy[int(n * training):int(n * (training + validation))]
    test_df = df_copy[int(n * (training + validation)):]
    return train_df, val_df, test_df


def data_transformation(scaler, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """ Performs data pre-processing according to scaler

    Parameters
    ----------
    scaler
    train_df
    val_df
    test_df

    Returns
    -------

    """
    # Must return a Pandas DataFrame
    train_df.loc[:, train_df.columns] = scaler.fit_transform(train_df.loc[:, train_df.columns])
    val_df.loc[:, val_df.columns] = scaler.transform(val_df.loc[:, val_df.columns])
    test_df.loc[:, test_df.columns] = scaler.transform(test_df.loc[:, test_df.columns])
    return train_df, val_df, test_df


def reg2class(df, n_classes):
    """ Regression to Classification problem
        Return the labels (0 to num_classes) & the mean per class
    Parameters
    ----------
    df
    n_classes

    Returns
    -------

    """
    label_encoder = LabelEncoder()
    # Create n_classes in the df
    classes = pd.cut(df, n_classes, retbins=True)
    # Class labels
    labels = label_encoder.fit_transform(classes[0])
    # Return also numeric value per class
    boundaries = classes[1]
    mean_class = []
    for i in range(len(boundaries) - 1):
        mean_class.append(0.5 * (boundaries[i] + boundaries[i + 1]))
    mean_class = np.array(mean_class)
    # Return only classes present in the dataset
    mean_class = mean_class[np.array(pd.cut(df, bins=n_classes).value_counts(sort=False)) != 0]
    return labels, mean_class


def class2num(y, mean_class):
    """ Convert classes to numeric output

    Parameters
    ----------
    y
    mean_class

    Returns
    -------

    """
    return mean_class[y]


def preprocessIoU(img, epsilon):
    levels = np.argmax(img, axis=0)
    # Column dilation of the pixels
    for column, level in enumerate(levels):
        for i in range(1, epsilon + 1):
            # Add white pixel on top
            img[min(level + i, 99), column] = 255
            # Add white pixel on bottom
            img[min(level - i, 99), column] = 255
    return img


def columnIoU(img_pred, img_gt, epsilon):
    # Preprocess IoU with epsilon
    img_gt = preprocessIoU(np.copy(img_gt), epsilon)
    img_pred = preprocessIoU(np.copy(img_pred), epsilon)
    # # Plotting for debug
    # fig, axes = plt.subplots(2, 1, figsize=(15, 6))
    # plt.suptitle('Test set: ground truth vs prediction', fontsize=16)
    # # Ground Truth
    # axes[0].imshow(img_gt, cmap="gray")
    # axes[0].set_title('Ground truth')
    # axes[0].axis("off")
    # # Prediction
    # axes[1].imshow(img_pred, cmap="gray")
    # axes[1].set_title('Prediction')
    # axes[1].axis("off")

    IoU = []
    for i in range(img_pred.shape[1]):
        # Get coordinates of bounding box
        column_gt = img_gt[:, i]
        column_pred = img_pred[:, i]
        # Get the rows whose value is white
        rows_gt = [i for i, _ in enumerate(column_gt) if _ == 255]
        rows_pred = [i for i, _ in enumerate(column_pred) if _ == 255]
        # determine the y-coordinates of the intersection rectangle (width is always 1)
        yA = max(min(rows_gt), min(rows_pred))
        yB = min(max(rows_gt), max(rows_pred))

        # compute the area of intersection rectangle
        interArea = abs(max((yB - yA), 0))
        if interArea == 0:
            IoU.append(0)
            continue
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxGt = abs(max(rows_gt) - min(rows_gt))
        boxPred = abs(max(rows_pred) - min(rows_pred))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxGt + boxPred - interArea)
        IoU.append(iou)
    # return the intersection over union value
    return np.mean(IoU)


def dtw(s, t):
    # n, m = len(s), len(t)
    # dtw_matrix = np.zeros((n+1, m+1))
    # for i in range(n+1):
    #     for j in range(m+1):
    #         dtw_matrix[i, j] = np.inf
    # dtw_matrix[0, 0] = 0
    #
    # for i in range(1, n+1):
    #     for j in range(1, m+1):
    #         cost = abs(s[i-1] - t[j-1])
    #         # take last min from a square box
    #         last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
    #         dtw_matrix[i, j] = cost + last_min
    # return dtw_matrix[n, m]
    distance, path = fastdtw(s, t, dist=euclidean)
    return distance
