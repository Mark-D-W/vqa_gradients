import mpi4py.MPI
import h5py
from quop_mpi.algorithm import qwoa
from quop_mpi import observable
import pandas as pd
import vqa_gradients
import os


qualities_df = pd.read_csv("qualities.csv")
qualities = qualities_df.values[:, 1]
system_size = len(qualities)



## PSR optimisation technique
alg = qwoa(system_size)
alg.set_qualities(observable.array, {"array": qualities})
alg.set_log("out/run_log", "vqa", action="a")

optimiser = vqa_gradients.Optimiser(R_W=2,
                                    R_Q=vqa_gradients.find_R_from_qualities(qualities))
alg.set_custom_optimiser(optimiser, optimiser_log=['fun','nfev','success'])

alg.benchmark(
    range(1, 6), 3, param_persist=True, filename="out/benchmark_vqa", save_action="w"
)
