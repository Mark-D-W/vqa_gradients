import mpi4py.MPI
import h5py
from quop_mpi.algorithm import qwoa
from quop_mpi import observable
import pandas as pd
import vqa_gradients

qualities_df = pd.read_csv("qwoa_qualities.csv")
qualities = qualities_df.values[:, 1]
system_size = len(qualities)


## Standard optimisation technique
alg = qwoa(system_size)
alg.set_qualities(observable.array, {"array": qualities})
alg.set_log("qwoa_portfolio_log", "qwoa", action="w")

alg.benchmark(
    range(1, 6), 3, param_persist=True, filename="qwoa_portfolio", save_action="w"
)




## PSR optimisation technique
alg = qwoa(system_size)
alg.set_qualities(observable.array, {"array": qualities})
alg.set_log("qwoa_portfolio_log_psr", "qwoa", action="w")

R_Q = vqa_gradients.Optimiser().find_R_from_qualities(qualities)
alg.set_custom_optimiser(vqa_gradients.Optimiser().call, {"R_Q":R_Q, "R_W":2})

alg.benchmark(
    range(1, 6), 3, param_persist=True, filename="qwoa_portfolio_psr", save_action="w"
)
