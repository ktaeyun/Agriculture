# tda_utils.py
import numpy as np
from gudhi import RipsComplex
from gudhi.wasserstein import wasserstein_distance

def compute_persistence_diagram_from_embedding(ts, dimension=3, delay=1, max_edge_length=2.0):
    ts = ts.flatten()
    ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-8)
    # Embedding
    N = len(ts)
    if N - (dimension - 1) * delay <= 0:
        raise ValueError("시계열 길이가 embedding에 충분하지 않음.")
    embedded = np.array([ts[i:N - (dimension - 1) * delay + i] for i in range(0, dimension * delay, delay)]).T
    # Rips filtration
    rips = RipsComplex(points=embedded, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=dimension)
    st.persistence()
    diag = st.persistence_intervals_in_dimension(0)
    return diag

def pd_distance(diag1, diag2, order=1., internal_p=2.):
    return wasserstein_distance(diag1, diag2, order=order, internal_p=internal_p)
