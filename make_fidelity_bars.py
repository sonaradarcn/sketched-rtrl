"""Regenerate fig_fidelity_bars directly from the published tab:fidelity numbers (5 seeds),
since the raw diagnostic-task JSONs are not local. Legend is placed above the axes so it never
overlaps the bars, and the a^nb^n task uses proper math rendering. Values are identical to
\\Cref{tab:fidelity} in the paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tasks = ["copy", "adding", "rotation", r"$a^nb^n$"]
# (label, means over the four tasks, stds)
data = [
    ("SK-RTRL r4",  [0.822, 0.884, 0.985, 0.979], [0.022, 0.019, 0.007, 0.004]),
    ("SK-RTRL r16", [0.856, 0.887, 0.981, 0.983], [0.039, 0.017, 0.003, 0.006]),
    ("SnAp-1",      [0.700, 0.606, 0.422, 0.425], [0.016, 0.021, 0.028, 0.089]),
    ("KF-RTRL",     [0.499, 0.501, 0.559, 0.450], [0.208, 0.050, 0.120, 0.034]),
    ("RFLO",        [0.343, 0.581, 0.157, 0.494], [0.024, 0.037, 0.033, 0.051]),
    ("UORO",        [0.040, 0.105, 0.059, 0.091], [0.022, 0.011, 0.021, 0.021]),
]
x = np.arange(len(tasks))
w = 0.8 / len(data)
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, (lab, vals, errs) in enumerate(data):
    ax.bar(x + i * w, vals, w, yerr=errs, label=lab, capsize=2)
ax.set_xticks(x + w * len(data) / 2)
ax.set_xticklabels(tasks)
ax.set_ylabel("Gradient cosine vs exact RTRL")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          frameon=False, columnspacing=1.0, handletextpad=0.4)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("../paper/figures/fig_fidelity_bars.pdf", bbox_inches="tight")
fig.savefig("../paper/figures/fig_fidelity_bars.png", dpi=300, bbox_inches="tight")
print("wrote fig_fidelity_bars (pdf+png)")
