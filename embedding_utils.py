# embedding_utils.py
import numpy as np

def time_delay_embedding(series, dimension=3, delay=1):
    series = series.flatten()
    N = len(series)
    if N - (dimension - 1) * delay <= 0:
        raise ValueError("Series too short for given dimension and delay")
    embedded = np.array([series[i:N - (dimension - 1) * delay + i] for i in range(0, dimension * delay, delay)]).T
    return embedded
