# tgmd.py
import numpy as np

def tuning_function(x, k=5):
    return 2 / (1 + np.exp(-k * (2 * x - 1)))

def compute_tgmd_matrix(topo_matrix, geo_matrix, k=5):
    # Normalize topo matrix
    td_min, td_max = np.min(topo_matrix), np.max(topo_matrix)
    td_scaled = (topo_matrix - td_min) / (td_max - td_min + 1e-8)

    def tuning_function(x, mode="soft", k=5):
        if mode == "original":
            return 2 / (1 + np.exp(-k * (2 * x - 1)))
        elif mode == "linear":
            return 0.5 + 0.5 * x
        elif mode == "sqrt":
            return np.sqrt(x)
        elif mode == "soft_exp":
            return 2 / (1 + np.exp(-1 * (2 * x - 1)))  # k=1 완화
        else:
            return x


    f_topo = tuning_function(td_scaled, mode="soft_exp")
    tgmd = f_topo * geo_matrix
    return tgmd

