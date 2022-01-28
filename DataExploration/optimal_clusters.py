import argparse
from BitbrainsUtils import *

# change path in BitbrainsUtils.py if needed

parser = argparse.ArgumentParser(
    description="Find optimal number of clusters of k-means")

parser.add_argument('-f', '--features', required=True,  nargs="+",
                    help='list of features used for clustering')

parser.add_argument('-m', '--model_path', default='./kmeans_models/',
                    help='Destination for model to be saved at')

parser.add_argument('-s', '--figure_path', default="Clustering/silhouette",
                    help='Destination for figures to be saved at')

args = parser.parse_args()

features = args.features
model_path = args.model_path
figure_path = args.figure_path

# Check model path
if not os.access(model_path, os.F_OK):
    os.mkdir(model_path)
if not os.access(model_path, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(model_path))
    exit()
else:
    print('models saved to {}'.format(model_path))

# Check figure path
figures_path = os.path.join(FIGURES_PATH,figure_path)
if not os.access(figures_path, os.F_OK):
    os.mkdir(figures_path)
if not os.access(figures_path, os.W_OK):
    print('Cannot write to {}, please fix it.'.format(figures_path))
    exit()
else:
    print('figures saved to {}'.format(figures_path))

print("--" * 40)
if __name__ == "__main__":
    print('Finding optimal k')
    print("--" * 40)
    sil_score = optimal_clusters(features, model_path)
    plot_silhouette(sil_score, savefig=figure_path)
