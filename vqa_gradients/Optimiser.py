# cython: annotation_typing = True
# cython: language_level = 3
import cython

from .Series import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.optimize import minimize

import numdifftools as nd


class Optimiser():
    def __init__(self, R_Q=None, R_W=None, n_qubits=None):
        self.objective=None
        self.R_Q=R_Q
        self.R_W=R_W
        self.n_qubits=n_qubits


    def __call__(self, func, param, **kwargs):
        self.objective = func
        for kw in kwargs:
            if kw=="R_Q":
                self.R_Q = kwargs[kw]
            elif kw=="R_W":
                self.R_W = kwargs[kw]
        return(
            minimize(self.objective,
                     param,
                     method="BFGS",
                     jac=self.__jac))




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

            #x = np.array([(2*np.pi*i)/(2*R+1) for i in range(0,2*R+1)])
            x = np.array([(2*np.pi*j)/(2*R+1) for j in range(0,2*R+1)])
            y = objective_i(x)
            s = Series(x,y)
            jac_vec[i] = s.gradient(param[i])
            #s.plot(function=objective_i)
        #print(f"PSR: jac={jac_vec}, numdifftools jac={nd.Jacobian(objective)(param)}")
        return(jac_vec)


