import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import config

df = pd.read_csv(config.PROJECT / "results" / "oof_predictions.csv")
fig_dir = config.PROJECT / "figures"
fig_dir.mkdir(exist_ok=True)
y = df["label"] == 1
types = ["engine", "elevator", "aileron", "rudder"]

# Figure 3: recall by fault type
raw, alarm, n = [], [], []
for t in types:
    m = y & (df["fault_type"] == t)
    raw.append(df.loc[m, "pred_raw"].mean())
    alarm.append(df.loc[m, "pred"].mean())
    n.append(int(m.sum()))
x = np.arange(len(types))
w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.2))
b1 = ax.bar(x - w / 2, raw, w, label="per-second prediction", color="#9ecae1")
b2 = ax.bar(x + w / 2, alarm, w, label="alarm (3 seconds in a row)", color="#08519c")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, "%.2f" % b.get_height(), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["%s\n(%d fault s)" % (t.capitalize(), k) for t, k in zip(types, n)])
ax.set_ylim(0, 1)
ax.set_ylabel("Recall (share of fault seconds detected)")
ax.set_title("Random forest, dev folds: recall by fault type")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(fig_dir / "fig3_recall_by_type.png", dpi=150)
plt.close(fig)

# Figure 4: confusion matrix (scikit-learn)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ConfusionMatrixDisplay.from_predictions(y, df["pred"], display_labels=["normal", "fault"],
                                        normalize="true", values_format=".2f",
                                        cmap="Blues", ax=ax, colorbar=False)
ax.set_xlabel("Predicted (alarm)")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix (share of each actual class)")
fig.tight_layout()
fig.savefig(fig_dir / "fig4_confusion_matrix.png", dpi=150)
plt.close(fig)

# Figure 5: alarm timeline, one row per dev flight with a fault
f = df[(df["fault_type"] != "none") & df["t_rel_fault"].notna()].copy()
order = f.groupby("flight").agg(ft=("fault_type", "first")).reset_index()
order["rank"] = order["ft"].map({t: i for i, t in enumerate(types)})
order = order.sort_values(["rank", "flight"]).reset_index(drop=True)
order["label"] = order["ft"] + " " + (order.groupby("ft").cumcount() + 1).astype(str)
fig, ax = plt.subplots(figsize=(9, 0.28 * len(order) + 1.6))
for i, row in order.iterrows():
    g = f[f["flight"] == row["flight"]]
    ax.hlines(i, g["t_rel_fault"].min(), g["t_rel_fault"].max(), color="#d9d9d9", lw=7, zorder=1)
    a = g[g["pred"]]
    ax.scatter(a["t_rel_fault"], [i] * len(a), marker="|", s=60, color="#cb181d", zorder=3, linewidths=1.5)
ax.axvline(0, color="black", lw=1.2, zorder=2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order["label"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Seconds relative to fault start (black line = fault starts, red tick = alarm)")
ax.set_title("Alarms per dev flight (out-of-fold predictions)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(fig_dir / "fig5_alarm_timeline.png", dpi=150)
plt.close(fig)

print("flights in timeline:", len(order))
print("saved fig3, fig4, fig5 in", fig_dir)
