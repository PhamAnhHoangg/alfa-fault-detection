"""Draws one fig7 cell as a scatter: engine dev rows, alt_error__mean5 against the label, both with the flight mean removed.

Run from the repo root:  python src/make_fig7_scatter.py
Writes figures/fig7_cell_scatter.png"""
import os
import numpy as np
import matplotlib.pyplot as plt
import evalkit

COL = "alt_error__mean5"
df = evalkit.load(cap=30)
g = df[(df["role"] != "test") & (df["fault_type"] == "engine")]
x = g[COL] - g.groupby("flight")[COL].transform("mean")
y = g["label"] - g.groupby("flight")["label"].transform("mean")
r = float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))
slope = float((x * y).sum() / (x ** 2).sum())  # least-squares line through 0, both sides are centred

fig, ax = plt.subplots(figsize=(7, 5))
for lab, colour, name in ((0, "tab:blue", "label 0 (before the fault)"), (1, "tab:red", "label 1 (fault on)")):
    m = g["label"] == lab
    ax.scatter(x[m], y[m], s=8, alpha=0.4, color=colour, label=name)
xs = np.linspace(x.min(), x.max(), 2)
ax.plot(xs, slope * xs, color="black", lw=1.5, label=f"line, slope {slope:.4f}")
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel(f"x~ = {COL} minus its flight mean")
ax.set_ylabel("y~ = label minus its flight mean")
ax.set_title(f"Fig7 cell {COL}, engine: r = {r:.2f} ({len(g)} rows, {g['flight'].nunique()} flights)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig(os.path.join("figures", "fig7_cell_scatter.png"), dpi=150)
print("saved figures/fig7_cell_scatter.png, r =", round(r, 3))
