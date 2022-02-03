import numpy as np

import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.optimize import minimize


class Series():
    def __init__(self, x, y):
        if np.mod(len(x), 2)!=1 or np.mod(len(y), 2)!=1 or len(x)!=len(y):
            raise Exception("x and y vectors must have length 2*R+1 where R is the number of unique eiginvalues of the function being reconstructed.")
        self.x = np.array(x)
        self.y = np.array(y, dtype=np.float64)
        self.R = int( (len(x) - 1)/2 )
        self.__create()


    def gradient(self, x):
        return( derivative(self.series, x, dx=1e-6) )


    def plot(self, function=None):
        range_x = 2*np.linspace(min(self.x), max(self.x), num=1000)
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

        def series(x):
            res = np.zeros((2*R+1,))
            for i in range(2*R+1):
                if i == 0:
                    res[i] = coef[0]
                elif np.mod(i, 2) == 0:
                    res[i] = coef[i] * np.sin(i/2 * x)
                elif np.mod(i, 2) == 1:
                    res[i] = coef[i] * np.cos((i+1)/2 * x)
            return(np.sum(res))

        self.series = series


    def __generate_fourier_term_matrix(self):
        x = self.x
        R = self.R
        series = np.zeros((2*R+1, 2*R+1))
        for i in range(2*R+1):
            for j in range(2*R+1):
                if i == 0:
                    series[j,i] = 1
                elif np.mod(i, 2) == 0:
                    series[j,i] = np.sin(i/2 * x[j])
                elif np.mod(i, 2) == 1:
                    series[j,i] = np.cos((i+1)/2 * x[j])
        #print(series)
        return(series)

