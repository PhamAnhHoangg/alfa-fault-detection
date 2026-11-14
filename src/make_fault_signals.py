"""Draws the 7 sensor signals around the fault, one panel per fault type (dev flights only).

Each signal is the per-second mean5 value, scaled per flight by its mean and standard deviation in the 30 s before
the fault, so 0 is that flight's normal level and 1 is one normal wiggle. The line is the median over flights.
Run from the repo root with PYTHONPATH=src:  python src/make_fault_signals.py
Writes figures/fig_signals_by_fault_type.png"""
import os
import matplotlib.pyplot as plt
import evalkit

SIGNALS = ["alt_error", "xtrack_error", "hud_altitude", "vel_lin_z", "roll_gap", "pitch_gap", "yaw_gap"]
TYPES = ["engine", "aileron", "rudder", "elevator"]
df = evalkit.load(cap=30)
dev = df[(df["role"] != "test") & df["fault_type"].isin(TYPES)].copy()
dev["s"] = dev["t_rel_fault"].round().astype(int)
dev = dev[dev["s"].between(-30, 30)]

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for ax, ft in zip(axes.flat, TYPES):
    g = dev[dev["fault_type"] == ft]
    for sig in SIGNALS:
        c = sig + "__mean5"
        pre = g[g["s"] < 0].groupby("flight")[c].agg(["mean", "std"])
        z = (g[c] - g["flight"].map(pre["mean"])) / g["flight"].map(pre["std"]).clip(lower=1e-6)
        ax.plot(z.groupby(g["s"]).median(), label=sig, lw=1.3)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(f"{ft} ({g['flight'].nunique()} dev flights)", fontsize=10)
    ax.set_ylim(-6, 6)
for ax in axes[1]:
    ax.set_xlabel("seconds from the fault start")
for ax in axes[:, 0]:
    ax.set_ylabel("median of scaled mean5")
axes[0, 0].legend(fontsize=7, ncol=2, loc="lower left")
fig.suptitle("Sensor signals around the fault, scaled by each flight's 30 s before the fault", fontsize=11)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig(os.path.join("figures", "fig_signals_by_fault_type.png"), dpi=150)
print("saved figures/fig_signals_by_fault_type.png")
print(dev.groupby("fault_type")["flight"].nunique().to_dict())
