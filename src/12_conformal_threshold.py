"""Conformal-style alarm threshold from out-of-fold forest scores (dev folds only).

For each fold, the threshold is the (1 - alpha) quantile of the scores of NORMAL seconds in the
OTHER folds, with the usual conformal correction. The fold being scored never sets its own
threshold. Then we check on that fold whether the promised false-alarm rate really holds."""
import math
import os, config
import numpy as np
import pandas as pd
import evalkit

ALPHAS = [0.05, 0.02, 0.01]
K_PERSIST = 3

oof = pd.read_csv(os.path.join(config.PROJECT, "results", "oof_predictions.csv"))
folds = sorted(oof["role"].unique())
print("folds:", folds, "| rows:", len(oof), "| normal seconds:", int((oof["label"] == 0).sum()))


def conformal_threshold(scores, alpha):
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


rows = []
for f in folds:
    test = oof[oof["role"] == f].reset_index(drop=True)
    cal = oof[(oof["role"] != f) & (oof["label"] == 0)]["proba"].to_numpy()
    normal = (test["label"] == 0).to_numpy()
    settings = [("fixed 0.5", 0.5, None)]
    settings += [(f"alpha={a}", conformal_threshold(cal, a), a) for a in ALPHAS]
    for name, thr, a in settings:
        raw = (test["proba"] >= thr) if a is None else (test["proba"] > thr)
        raw = raw.to_numpy()
        res = evalkit.evaluate(test, raw, k_persist=K_PERSIST)
        res.update(fold=f, setting=name, threshold=thr, raw_fa_rate=float(raw[normal].mean()))
        rows.append(res)

out = pd.DataFrame(rows)
os.makedirs(config.PROJECT / "results", exist_ok=True)
out.to_csv(os.path.join(config.PROJECT, "results", "conformal_threshold.csv"), index=False)

order = ["fixed 0.5"] + [f"alpha={a}" for a in ALPHAS]
metrics = ["threshold", "raw_fa_rate", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min", "med_delay_s"]
print("\nMean over the 5 folds (alarm rule: 3 seconds in a row):")
print(out.groupby("setting")[metrics].mean().loc[order].round(3).to_string())

print("\nSanity check: fixed 0.5 mean F1 =",
      round(out[out["setting"] == "fixed 0.5"]["f1"].mean(), 2), "(earlier result: 0.51)")

print("\nPromise check: share of normal seconds that raised a raw alarm, per fold (target = alpha):")
print(out.pivot(index="fold", columns="setting", values="raw_fa_rate")[order].round(3).to_string())

print("\nThreshold used, per fold:")
print(out.pivot(index="fold", columns="setting", values="threshold")[order].round(3).to_string())
