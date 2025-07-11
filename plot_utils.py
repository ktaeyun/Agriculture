# plot_utils.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_clusters(X, labels, y_true=None):
    X_flat = X.reshape(len(X), -1)
    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(X_flat)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    if y_true is not None:
        plt.title("TGMD Clustering (True classes available)")
    else:
        plt.title("TGMD Clustering")
    plt.show()