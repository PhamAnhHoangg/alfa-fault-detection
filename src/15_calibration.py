"""Self-check 2: are the model scores trustworthy as probabilities? (dev folds, seed 0, out-of-fold)"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

oof = pd.read_csv(os.path.join("results", "oof_models.csv"))
oof = oof[oof["seed"] == 0]
prev = oof[oof["model"] == "forest"]["label"].mean()
print("Fault share:", round(prev, 3),
      "| Brier score of always predicting that share:", round(prev * (1 - prev), 3))

colors = {"forest": "#2a78d6", "catboost": "#eb6834", "mlp": "#1baf7a"}
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect calibration")

for model in ["forest", "catboost", "mlp"]:
    d = oof[oof["model"] == model]
    y, p = d["label"].to_numpy(), d["proba"].to_numpy()
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="uniform")
    edges = np.linspace(0, 1, 11)
    counts = np.bincount(np.searchsorted(edges[1:-1], p), minlength=10)
    counts = counts[counts > 0]
    brier = brier_score_loss(y, p)
    table = pd.DataFrame({"mean_score": mean_p, "fault_rate": frac, "seconds": counts})
    print(f"\n{model}: Brier score {brier:.3f}")
    print(table.round(3).to_string(index=False))
    ax.plot(mean_p, frac, marker="o", color=colors[model], label=f"{model} (Brier {brier:.3f})")

ax.set_xlabel("Mean model score in the bin")
ax.set_ylabel("Share of seconds in the bin that were real faults")
ax.set_title("Calibration on held-out folds")
ax.legend(loc="upper left", frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
os.makedirs("figures", exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join("figures", "fig9_calibration.png"), dpi=150)
print("\nsaved figures\\fig9_calibration.png")
