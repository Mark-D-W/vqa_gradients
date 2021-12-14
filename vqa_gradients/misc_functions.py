# cython: annotation_typing = True
# cython: language_level = 3
import cython

import numpy as np


def complete_graph(n):
    return np.ones((n,n)) - np.diag(np.ones(n))
