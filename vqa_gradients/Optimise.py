# cython: annotation_typing = True
# cython: language_level = 3
import cython

from .Series import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.optimize import minimize


class Optimise():
    def __init__(self):
        self.objective=None
        self.R_Q=None
        self.R_W=None

        
    def optimiser(self, func, param, **kwargs):
        self.objective = func
        for kw in kwargs:
            if kw=="R_Q":
                self.R_Q = kwargs[kw]
            elif kw=="R_W":
                self.R_W = kwargs[kw]

        psr_gradient = self.__jac(param)
        fd_gradient = [self.__partial_derivative(self.objective,param,i) for i,v in enumerate(param)]
        #print(f"The psr gradient is: {psr_gradient} at {param}\nThe fd gradient is: {fd_gradient} at {param}\n")

        return(
            minimize(self.objective,
                     param,
                     method="BFGS",
                     jac=self.__jac))


    def find_R_from_qualities(self, qualities):
        R_Q = self.__num_unique_positive_differences(qualities)
        return(R_Q)


    def __partial_derivative(self, func, param, i):
        wraps = lambda x: func([val if idx!=i else x for idx,val in enumerate(param)])
        return derivative(wraps, param[i], dx=1e-6)



    def __num_unique_positive_differences(self, array, epsilon=1e-6):
        diffs = [0] #There is always 0 difference
        array.sort()
        for idx_i,val_i in enumerate(array):
            for idx_j,val_j in enumerate(array[idx_i:]):
                diff = np.real(val_j - val_i)
                if diff>epsilon:
                    diffs.append(diff)
        return(len(np.unique(diffs)))


    def __jac(self, param):
        objective = self.objective
        R_Q = self.R_Q
        R_W = self.R_W
        jac_vec = np.empty((len(param),))

        for i in range(len(param)):
            if i%2==0:
                R = R_Q  # Phase shift operator
            else:
                R = R_W  # Walk operator
            objective_i = np.vectorize( lambda x: objective([val if idx!=i else x for idx,val in enumerate(param)]) )

            #x = np.array([(2*np.pi*i)/(2*R+1) for i in range(-R,R+1)])
            x = np.array([(2*np.pi*i)/(2*R+1) for i in range(0,2*R+1)])
            y = objective_i(x)
            s = Series(x,y)
            if i%2==0:
                jac_vec[i] = s.gradient(param[i])
            else:
                #jac_vec[i] = derivative(objective_i, param[i], dx=1e-6)
                jac_vec[i] = s.gradient(param[i])
            #s.plot(function=objective_i)
        return(jac_vec)


