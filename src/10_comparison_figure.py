"""Figure 8: model comparison (reads results/model_comparison.csv)."""
import os, config
import pandas as pd
import matplotlib.pyplot as plt

out = pd.read_csv(os.path.join(config.PROJECT, "results", "model_comparison.csv"))
models = ["forest", "catboost", "mlp"]
colors = {"forest": "#4c72b0", "catboost": "#dd8452", "mlp": "#55a868"}

# one value per model and fold: the mean over the 5 seeds
fm = out.groupby(["model", "fold"])[["f1", "rec_eng", "fa_s_per_min"]].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# left: F1 per fold
f1 = fm.pivot(index="fold", columns="model", values="f1")[models]
for fold, row in f1.iterrows():
    ax1.plot(range(3), row.values, color="lightgray", marker="o", zorder=1)
    ax1.text(2.08, row.values[2], fold, fontsize=7, va="center")
for i, m in enumerate(models):
    ax1.scatter([i] * len(f1), f1[m], color=colors[m], s=45, zorder=2)
    ax1.scatter(i - 0.17, f1[m].mean(), color="black", marker="D", s=70, zorder=3)
ax1.set_xticks(range(3))
ax1.set_xticklabels(models)
ax1.set_xlim(-0.4, 2.4)
ax1.set_ylim(0.2, 0.9)
ax1.set_ylabel("F1 on the held-out fold (mean of 5 seeds)")
ax1.set_title("F1 per fold")
ax1.text(0.02, 0.02, "gray lines join the same fold; black diamond = mean",
         transform=ax1.transAxes, fontsize=8)

# right: recall vs false alarms
for m in models:
    d = fm[fm["model"] == m]
    ax2.scatter(d["fa_s_per_min"], d["rec_eng"], color=colors[m], alpha=0.35, s=35)
    ax2.scatter(d["fa_s_per_min"].mean(), d["rec_eng"].mean(), color=colors[m],
                s=140, edgecolor="black", label=m, zorder=3)
ax2.set_xlabel("false-alarm seconds per minute of normal flight")
ax2.set_ylabel("engine-fault recall")
ax2.set_title("Engine recall vs false alarms (model means)")
ax2.legend(title="big dot = mean, small dots = folds")

fig.tight_layout()
os.makedirs(config.PROJECT / "figures", exist_ok=True)
path = os.path.join(config.PROJECT, "figures", "fig8_model_comparison.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print("saved", path)
