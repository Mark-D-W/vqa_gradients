import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd

plt.rcParams["font.size"] = 16

figure_size = (5, 4)

qwoa_benchmark_df = pd.read_csv("qwoa_portfolio_log.csv")
qwoa_depth_min = qwoa_benchmark_df["ansatz_depth"].min()
qwoa_depth_max = qwoa_benchmark_df["ansatz_depth"].max()

qwoa_benchmark_df_psr = pd.read_csv("qwoa_portfolio_log_psr.csv")
qwoa_depth_min_psr = qwoa_benchmark_df_psr["ansatz_depth"].min()
qwoa_depth_max_psr = qwoa_benchmark_df_psr["ansatz_depth"].max()


depths = []
qwoa_fun_mean = []
for depth in range(1, 6):
    qwoa_fun_mean.append(
        qwoa_benchmark_df[qwoa_benchmark_df["ansatz_depth"] == depth]["fun"].mean()
    )

plt.figure(figsize=figure_size)

plt.plot(
    qwoa_benchmark_df["ansatz_depth"],
    qwoa_benchmark_df["fun"],
    "o",
    markersize=6,
    color="tab:blue",
    label="QWOA",
)



depths = []
qwoa_fun_mean_psr = []
for depth in range(1, 6):
    qwoa_fun_mean_psr.append(
        qwoa_benchmark_df_psr[qwoa_benchmark_df_psr["ansatz_depth"] == depth]["fun"].mean()
    )

plt.figure(figsize=figure_size)

plt.plot(
    qwoa_benchmark_df_psr["ansatz_depth"],
    qwoa_benchmark_df_psr["fun"],
    "o",
    markersize=6,
    color="tab:orange",
    label="QWOA PSR",
)


plt.plot(list(range(1, 6)), qwoa_fun_mean_psr, "--", color="tab:orange")
plt.plot(list(range(1, 6)), qwoa_fun_mean, "--", color="tab:blue")

plt.xticks([i for i in range(qwoa_depth_min, qwoa_depth_max + 1)])
plt.xlabel("depth (D)")
plt.ylabel("quality expectation value")

plt.grid(which="major", linestyle="--")
plt.legend(fontsize="medium", framealpha=1, borderpad=0.1)

plt.tight_layout()
plt.savefig("portfolio_rebalancing", dpi=200)
