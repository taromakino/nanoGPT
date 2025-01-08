import os
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 16


num_seeds = 3
shared_layers = [
    "None",
    "{6, 7}",
    "{7, 8}"
]
dirs = [os.path.join("results", f"shared_layers={item}") for item in shared_layers]


def get_test_acc(dir, seed):
    file_path = os.path.join(dir, f"seed={seed}.txt")
    with open(file_path, "r") as file:
        return float(file.read())


fig, ax = plt.subplots(1, 1, figsize=(6, 4))

means, sds = [], []
for dir in dirs:
    test_accs = [get_test_acc(dir, seed) for seed in range(num_seeds)]
    means.append(np.mean(test_accs))
    sds.append(np.std(test_accs))

ax.errorbar(np.arange(len(shared_layers)), means, yerr=sds, marker="o", capsize=3, label="Without layer embedding")

means, sds = [], []
for dir in dirs:
    test_accs = [get_test_acc(dir + "_wle", seed) for seed in range(num_seeds)]
    means.append(np.mean(test_accs))
    sds.append(np.std(test_accs))

ax.errorbar(np.arange(len(shared_layers)) + 0.05, means, yerr=sds, marker="o", capsize=3, label="With layer embedding")

ax.set_xlabel("Shared layers")
ax.set_xticks(range(len(shared_layers)))
ax.set_xticklabels(shared_layers)

ax.set_ylabel("bpc")

fig.legend(loc="lower center", bbox_to_anchor=[0.5, 0.])
fig.tight_layout()
fig.subplots_adjust(bottom=0.375)

fig.savefig("fig.jpeg")