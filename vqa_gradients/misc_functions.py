import numpy as np


def complete_graph(n):
    return np.ones((n,n)) - np.diag(np.ones(n))
