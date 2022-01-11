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


## Standard optimisation technique
alg = qwoa(system_size)
alg.set_qualities(observable.array, {"array": qualities})
alg.set_log("out/run_log", "standard", action="w")

alg.benchmark(
    range(1, 6), 3, param_persist=True, filename="out/benchmark_standard", save_action="w"
)
