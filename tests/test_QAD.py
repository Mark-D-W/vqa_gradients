from vqa_gradients import QAD
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping

local_minima = []
def append_local_minima(x, f, accept):
    local_minima.append([x, f, accept])

def save_local_minima():
    with open("local_minima.log", "w") as infile:
        infile.write(
            str(local_minima).replace("], [", "]\n[")
        )
    

def QAD_test():
    func = lambda x: np.sin(x[0]) + x[2]**2 + 7*x[2] - 2*np.sin(x[1])
    param = np.random.random(3)
    qad = QAD(optimiser=basinhopping, optimiser_args={"callback":append_local_minima})
    res = qad(func, param)
    save_local_minima()
    print(f"QAD minima found={res}")
    print(f"Scipy minima found={minimize(func, param)['fun']}")



QAD_test()
