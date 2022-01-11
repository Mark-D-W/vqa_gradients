import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd

plt.rcParams["font.size"] = 16

figure_size = (5, 4)

log = pd.read_csv("out/run_log.csv")


standard_log_file = log[log["label"]=="standard"]
standard_depth_min = standard_log_file["ansatz_depth"].min()
standard_depth_max = standard_log_file["ansatz_depth"].max()

vqa_log_file = log[log["label"]=="vqa"]
vqa_depth_min = vqa_log_file["ansatz_depth"].min()
vqa_depth_max = vqa_log_file["ansatz_depth"].max()


depths = []
standard_fun_mean = []
for depth in range(1, 6):
    standard_fun_mean.append(
        standard_log_file[standard_log_file["ansatz_depth"] == depth]["fun"].mean()
    )

plt.figure(figsize=figure_size)

plt.plot(
    standard_log_file["ansatz_depth"],
    standard_log_file["fun"],
    "o",
    markersize=6,
    color="tab:blue",
    label="Standard",
)



depths = []
vqa_fun_mean = []
for depth in range(1, 6):
    vqa_fun_mean.append(
        vqa_log_file[vqa_log_file["ansatz_depth"] == depth]["fun"].mean()
    )

plt.plot(
    vqa_log_file["ansatz_depth"],
    vqa_log_file["fun"],
    "o",
    markersize=6,
    color="tab:orange",
    label="vqa_gradients",
)


plt.plot(list(range(1, 6)), standard_fun_mean, "--", color="tab:blue")
plt.plot(list(range(1, 6)), vqa_fun_mean, "--", color="tab:orange")

plt.xticks([i for i in range(standard_depth_min, standard_depth_max + 1)])
plt.xlabel("depth (D)")
plt.ylabel("quality expectation value")

plt.grid(which="major", linestyle="--")
plt.legend(fontsize="medium", framealpha=1, borderpad=0.1)

plt.tight_layout()
plt.savefig("out/plot1", dpi=200)
