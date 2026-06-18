"""Pilot figure substantiating the off-diagonal-residual claim (Section 3).
Left: fraction of total influence-matrix Frobenius mass carried by the off-diagonal
residual J - S over training, per task. Right: cumulative top-k singular-value mass of
the residual at convergence, showing approximate low rank. Outputs PDF (vector) + PNG.
Data: results/m1_spectrum_<task>.json (fields res_frac_of_J, top{1,4,8,16,32,64}).
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
TASKS = {"copy": "copy", "adding": "adding", "rotation": "rotation", "anbn": "$a^nb^n$"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.6))
ks = [1, 4, 8, 16, 32, 64]
for f in sorted(glob.glob("results/m1_spectrum_*.json")):
    d = json.load(open(f))
    t = d["args"]["task"]
    lbl = TASKS.get(t, t)
    recs = [r for r in d["records"] if r.get("res_frac_of_J", 0) > 0]
    if not recs:
        continue
    # left: residual mass fraction over training
    axL.plot([r["step"] for r in recs], [r["res_frac_of_J"] for r in recs],
             marker="o", ms=3, label=lbl)
    # right: cumulative top-k mass at the last logged point
    last = recs[-1]
    axR.plot(ks, [last[f"top{k}"] for k in ks], marker="s", ms=4, label=lbl)

axL.set_xlabel("online step")
axL.set_ylabel(r"$\|J-S\|_F\,/\,\|J\|_F$")
axL.set_title("Off-diagonal residual mass fraction")
axL.set_ylim(0, 1)
axL.grid(alpha=0.3)
axL.legend(fontsize=8)
axR.set_xlabel("retained rank $k$")
axR.set_ylabel("cumulative singular mass")
axR.set_title("Residual is approximately low rank")
axR.set_xscale("log", base=2)
axR.set_ylim(0, 1.02)
axR.grid(alpha=0.3)
axR.legend(fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_pilot_residual.{ext}", dpi=300)
print("saved fig_pilot_residual.pdf / .png")
