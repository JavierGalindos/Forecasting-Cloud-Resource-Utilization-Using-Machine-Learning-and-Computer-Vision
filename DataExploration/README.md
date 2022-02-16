# Dataset exploration

## Bitbrains
### BitbrainsUtil.py
Utilities for the Bitbrains dataset
### Notebook analyzing the dataset
[Bitbrains data exploration](DataExploration/BitbrainsDataset.ipynb)

### ClusteringUtils.py
Utilities for the clustering in the Bitbrains dataset
### optimal_cluster.py
Finds the optimal number of clustering using silhouette coefficient.
Generates a figure.

To run:
```
python optimal_clusters.py -f <features used for clustering> -m path to save the models> -s <path to save the figure>
```

Example:
```
python3 optimal_clusters.py -f 'Memory usage [KB]' -m './k_means_memory/' -s 'Clustering/silhouette_memory'
```