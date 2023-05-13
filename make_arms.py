import numpy as np

def generate_data(n, n_arms, locs, scales):
    np.random.seed(1)
    qs = np.random.normal(loc=locs, scale=scales, size=(n_arms))
    d = np.array([np.random.normal(loc=qs[a], scale=np.sqrt(scales), size=n) for a in range(n_arms)])
    return d

