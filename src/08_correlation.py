"""Correlation check on dev flights.
1) Which signals move with the fault label (compared inside each flight).
2) Which signals duplicate each other (redundancy)."""
import os, config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evalkit

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"].copy()
sym, feats = evalkit.feature_lists(dev)
cols = [c for c in sym if not c.startswith("rc_out")]
print("dev rows:", len(dev), "| flights:", dev["flight"].nunique(), "| features:", len(cols))
print("fault_type values:", sorted(dev["fault_type"].dropna().astype(str).unique()))


def within_flight_corr(g):
    # subtract each flight's own average, then correlate with the label
    x = g[cols] - g.groupby("flight")[cols].transform("mean")
    y = g["label"] - g.groupby("flight")["label"].transform("mean")
    return x.corrwith(y)


types = ["engine", "elevator", "aileron", "rudder"]
rows = {}
for t in types:
    g = dev[dev["fault_type"] == t]
    if g.empty:
        print("\nno rows for fault type:", t)
        continue
    rows[t] = within_flight_corr(g)
    top = rows[t].reindex(rows[t].abs().sort_values(ascending=False).index).head(10)
    print(f"\n{t}: flights={g['flight'].nunique()}, fault rows={int(g['label'].sum())}")
    print("top 10 by |within-flight r| with the label")
    print(top.round(2).to_string())

table = pd.DataFrame(rows)

# redundancy between features (plain correlation over all dev rows)
cm = dev[cols].corr()
pairs = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = cm.iloc[i, j]
        if abs(r) > 0.9:
            pairs.append((cols[i], cols[j], r))
pairs.sort(key=lambda p: -abs(p[2]))
print("\nfeature pairs with |r| > 0.9:", len(pairs))
for a, b, r in pairs:
    print(f"  {r:+.2f}  {a}  <->  {b}")

os.makedirs(config.PROJECT / "figures", exist_ok=True)

# fig7: feature vs label, per fault type
data = table.fillna(0).to_numpy()
fig, ax = plt.subplots(figsize=(6.5, 9))
im = ax.imshow(data, cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(table.shape[1]))
ax.set_xticklabels(table.columns)
ax.set_yticks(range(table.shape[0]))
ax.set_yticklabels(table.index, fontsize=8)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="within-flight r with the fault label")
ax.set_title("Which signals move with the fault (inside each flight)")
fig.tight_layout()
fig.savefig(os.path.join(config.PROJECT, "figures", "fig7_label_correlation.png"), dpi=150)
plt.close(fig)

# fig6: feature vs feature
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(cols, fontsize=6)
fig.colorbar(im, ax=ax, label="correlation r")
ax.set_title("Which features duplicate each other (dev rows)")
fig.tight_layout()
fig.savefig(os.path.join(config.PROJECT, "figures", "fig6_feature_correlation.png"), dpi=150)
plt.close(fig)
print("\nsaved figures\\fig6_feature_correlation.png and figures\\fig7_label_correlation.png")
