# geo_utils.py
from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean
import numpy as np

def dtw_distance(ts1, ts2):
    return dtw(ts1.flatten(), ts2.flatten())

def euclidean_distance(ts1, ts2):
    ts1 = ts1.flatten()
    ts2 = ts2.flatten()
    return euclidean(ts1, ts2)

