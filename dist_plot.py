import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.stats as stats

x_ticks_0 = np.arange(0.76, 0.81, 0.01)
x_ticks_1 = np.arange(0.92, 0.94, 0.01)

with open(Path("runs_dist/res_dist.csv"), "r") as f:
    res = {}
    for line in f.readlines():
        l = line.split(",")
        res[l[0] + l[1]] = list(map(float, l[2:]))

fig, axs = plt.subplots(2, 1, figsize=(6, 5))
# auc
bp1 = axs[0].boxplot(
    [res["sgcn2auc"], res["sgcn1pauc"], res["sgcn1auc"]],
    labels=["SGCN-2", "SGCN-1+", "SGCN-1"],
    vert=False,
)
axs[0].set_xlabel("AUC")
axs[0].legend([bp1["medians"][0]], ["median"])
axs[0].set_xticks(x_ticks_0)


# f1
bp2 = axs[1].boxplot(
    [res["sgcn2f1"], res["sgcn1pf1"], res["sgcn1f1"]],
    labels=["SGCN-2", "SGCN-1+", "SGCN-1"],
    vert=False,
)
axs[1].set_xlabel("F1")
axs[1].legend([bp2["medians"][0]], ["median"])
axs[1].set_xticks(x_ticks_1)
# plt.show()
plt.tight_layout()
fig.savefig("dist_plot.png")

f_auc, p_auc = stats.f_oneway(res["sgcn2auc"], res["sgcn1auc"], res["sgcn1pauc"])
f_f1, p_f1 = stats.f_oneway(res["sgcn2f1"], res["sgcn1f1"], res["sgcn1pf1"])
print(f"{p_auc=}. Can we reject null hypotheses? {p_auc < 0.05}")
print(f"{p_f1=}. Can we reject null hypotheses? {p_f1 < 0.05}")

print("### STD DEVS AND MEAN ###")
for name, results in res.items():
    print(f"{name}:\tstd: {np.std(results)},\tmean:{np.mean(results)}")

f_auc_1_1p, p_auc_1_1p = stats.f_oneway(res["sgcn1auc"], res["sgcn1pauc"])
f_auc_2_1p, p_auc_2_1p = stats.f_oneway(res["sgcn2auc"], res["sgcn1pauc"])
f_auc_2_1, p_auc_2_1 = stats.f_oneway(res["sgcn2auc"], res["sgcn1auc"])
f_f1_1_1p, p_f1_1_1p = stats.f_oneway(res["sgcn1f1"], res["sgcn1pf1"])
f_f1_2_1p, p_f1_2_1p = stats.f_oneway(res["sgcn2f1"], res["sgcn1pf1"])
f_f1_2_1, p_f1_2_1 = stats.f_oneway(res["sgcn2f1"], res["sgcn1f1"])

for f_value, p_value, string in [
    (f_auc_1_1p, p_auc_1_1p, "AUC, SGCN-1, SGCN-1+"),
    (f_auc_2_1p, p_auc_2_1p, "AUC, SGCN-2, SGCN-1+"),
    (f_auc_2_1, p_auc_2_1, "AUC, SGCN-2, SGCN-1"),
    (f_f1_1_1p, p_f1_1_1p, "F1, SGCN-1, SGCN-1+"),
    (f_f1_2_1p, p_f1_2_1p, "F1, SGCN-2, SGCN-1+"),
    (f_f1_2_1, p_f1_2_1, "F1, SGCN-2, SGCN-1"),
]:
    print(f"For {string}: {f_value=}, {p_value=}. Can we reject null hypotheses? {p_value < 0.05}")
