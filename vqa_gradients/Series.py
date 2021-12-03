import numpy as np

import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.optimize import minimize


class Series():
    def __init__(self, x, y):
        if np.mod(len(x), 2)!=1 or np.mod(len(y), 2)!=1 or len(x)!=len(y):
            raise Exception("x and y vectors must have length 2*R+1 where R is the number of unique eiginvalues of the function being reconstructed.")
        self.x = x
        self.y = y
        self.R = int( (len(x) - 1)/2 )
        self.__create()


    def gradient(self, x):
       return( derivative(self.series, x, dx=1e-6) )


    def plot(self, function=None):
       range_x = np.linspace(min(self.x), max(self.x), num=100)
       series_y = np.vectorize(self.series)(range_x)
       plt.scatter(self.x, self.y, color="red", zorder=3, label="Sampled points")
       plt.plot(range_x, series_y, linestyle="dashed", linewidth=3, color="blue", zorder=2, label="Reconstructed function")
       if function!=None:
          function_y = np.vectorize(function)(range_x)
          plt.plot(range_x,function_y, linewidth=2, color="black", zorder=1, label="Original function")
       plt.legend()
       plt.show()
       plt.close()


    def __create(self):
        fourier_term_mat = self.__generate_fourier_term_matrix()
        coef = np.linalg.solve(fourier_term_mat, self.y)
        R = self.R
        Omega = np.arange(1,R+1)

        def series(x):
            res = np.zeros((2*R+1,))
            for i in range(2*R+1):
                if i == 0:
                    res[i] = coef[0]
                elif np.mod(i, 2) == 0:
                    res[i] = coef[i] * np.sin(Omega[int(i/2)-1] * x)
                elif np.mod(i, 2) == 1:
                    res[i] = coef[i] * np.cos(Omega[int(((i-1)/2)-1)] * x)
            return(np.sum(res))

        self.series = series


    def __generate_fourier_term_matrix(self):
        x = self.x
        R = self.R
        Omega = np.arange(1,R+1)
        series = np.zeros((2*R+1, 2*R+1))
        for i in range(2*R+1):
            for j in range(2*R+1):
                if i == 0:
                    series[j,i] = 1
                elif np.mod(i, 2) == 0:
                    series[j,i] = np.sin(Omega[int(i/2)-1] * x[j])
                elif np.mod(i, 2) == 1:
                    series[j,i] = np.cos(Omega[int((i-1)/2-1)] * x[j])
        return(series)


    
  

    
def psr_jac(param, objective, R):
    jac_vec = np.empty((len(param),))
    for i in range(len(param)):
        objective_i = np.vectorize( lambda x: objective([var if var!=param[i] else x for var in param]) )
        x = np.linspace(1,10,num=2*R+1)
        y = objective_i(x)
        jac_vec[i] = Series(x,y).gradient(param[i])
    return(jac_vec)


def psr_optimise(func, param, **kwarg):
    def objective2(param, *args):
        return(func(param))
    return(
        minimize(objective2,
                 param,
                 jac=psr_jac,
                 args=(func, 10)))
