from vqa_gradients import QAD
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping


def QAD_test():
    func = lambda x: np.sin(x[0]) + x[2]**2 - 2*np.sin(x[1])
    param = np.random.random(3)
    qad = QAD(optimiser=basinhopping)
    res = qad(func, param)
    print(f"res={res}")
    print(f"actual minima={minimize(func, param)['fun']}")


QAD_test()
