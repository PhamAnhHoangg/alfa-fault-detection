"""Conformal-style alarm threshold for forest, CatBoost and MLP (dev folds only).

Same method as 12_conformal_threshold.py, but on the out-of-fold probabilities saved by
09_model_comparison.py (results/oof_models.csv). Threshold for a fold = (1 - alpha) quantile of the
NORMAL-second scores of the other folds, same model and same seed. Results are averaged over folds
and seeds."""
import math
import os
import numpy as np
import pandas as pd
import evalkit

ALPHAS = [0.05, 0.02, 0.01]
K_PERSIST = 3

oof = pd.read_csv(os.path.join("results", "oof_models.csv"))
models = sorted(oof["model"].unique())
seeds = sorted(oof["seed"].unique())
print("rows:", len(oof), "(expected 49185) | models:", models, "| seeds:", seeds)


def conformal_threshold(scores, alpha):
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


rows = []
for model in models:
    for seed in seeds:
        d = oof[(oof["model"] == model) & (oof["seed"] == seed)]
        for f in sorted(d["role"].unique()):
            test = d[d["role"] == f].reset_index(drop=True)
            cal = d[(d["role"] != f) & (d["label"] == 0)]["proba"].to_numpy()
            normal = (test["label"] == 0).to_numpy()
            settings = [("fixed 0.5", 0.5, None)]
            settings += [(f"alpha={a}", conformal_threshold(cal, a), a) for a in ALPHAS]
            for name, thr, a in settings:
                raw = (test["proba"] >= thr) if a is None else (test["proba"] > thr)
                raw = raw.to_numpy()
                res = evalkit.evaluate(test, raw, k_persist=K_PERSIST)
                res.update(model=model, seed=seed, fold=f, setting=name, threshold=thr,
                           raw_fa_rate=float(raw[normal].mean()))
                rows.append(res)

out = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
out.to_csv(os.path.join("results", "conformal_models.csv"), index=False)

order = ["fixed 0.5"] + [f"alpha={a}" for a in ALPHAS]
metrics = ["threshold", "raw_fa_rate", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min", "med_delay_s"]
per_seed = out.groupby(["model", "setting", "seed"])[metrics].mean()
table = per_seed.groupby(["model", "setting"])[metrics].mean()
table = table.reindex(pd.MultiIndex.from_product([models, order], names=["model", "setting"]))
print("\nMean over 5 folds, then 5 seeds (alarm rule: 3 seconds in a row):")
print(table.round(3).to_string())

print("\nPromise check: raw false-alarm rate per fold (mean over seeds). Target = alpha.")
pc = out[out["setting"] != "fixed 0.5"].groupby(["model", "setting", "fold"])["raw_fa_rate"].mean()
print(pc.unstack("fold").round(3).to_string())
