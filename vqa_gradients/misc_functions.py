# cython: annotation_typing = True
# cython: language_level = 3
import cython

import numpy as np
from scipy.misc import derivative



def partial_derivative(func, param, i, dx=1e-6, n=1):
    wraps = lambda x: func([val if idx!=i else x for idx,val in enumerate(param)])
    return derivative(wraps, param[i], dx=dx, n=n)


def jacobian(func, param):
    return np.array([partial_derivative(func, param, i)
                     for i in range(len(param))])



def complete_graph(n):
    return np.ones((n,n)) - np.diag(np.ones(n))


def __num_unique_positive_differences(array, epsilon=1e-6):
    diffs = [0] #There is always 0 difference
    array.sort()
    for idx_i,val_i in enumerate(array):
        for idx_j,val_j in enumerate(array[idx_i:]):
            diff = np.real(val_j - val_i)
            if diff>epsilon:
                diffs.append(diff)
    return(len(np.unique(diffs)))


def find_R_from_qualities(qualities):
    R_Q = __num_unique_positive_differences(qualities)
    return(R_Q)

