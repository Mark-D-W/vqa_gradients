import mpi4py.MPI
import h5py
from quop_mpi.algorithm import qwoa
from quop_mpi import observable
from quop_mpi import optimiser
import pandas as pd
import vqa_gradients
import os


qualities_df = pd.read_csv("qualities.csv")
qualities = qualities_df.values[:, 1]
system_size = len(qualities)


## Standard optimisation technique
alg = qwoa(system_size)
alg.set_qualities(observable.array, {"array": qualities})
alg.set_log("out/run_log", "standard", action="w")




#optimiser = optimiser.ScipyOptimiser(method="BFGS")
optimiser = optimiser.NloptOptimiser()

alg.set_optimiser(optimiser, optimiser_log=['fun','nfev','success'])
#alg.set_optimiser("scipy", {"method":"BFGS"}, ['fun','nfev','success'])
#alg.set_optimiser("nlopt", optimiser_log=['fun','nfev','success'])




alg.benchmark(
    range(1, 6), 3, param_persist=True, filename="out/benchmark_standard", save_action="w"
)
