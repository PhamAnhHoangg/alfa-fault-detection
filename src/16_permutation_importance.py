"""Self-check 3: which features does the forest rely on? (dev folds only, seed 0)

Permutation importance = drop in average precision on a held-out fold when one feature's values are
shuffled. Duplicated features share credit, so a group test shuffles related columns together."""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
import evalkit

SEED = 0
N_REPEATS = 10

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"].reset_index(drop=True)
sym, feats = evalkit.feature_lists(dev)
cols = [c for c in sym if not c.startswith("rc_out")]
folds = sorted(dev["role"].unique())
print("dev rows:", len(dev), "| features:", len(cols))

alt = [i for i, c in enumerate(cols) if "alt_error" in c]
hud = [i for i, c in enumerate(cols) if "hud_altitude" in c]
print("alt_error columns:", [cols[i] for i in alt])
print("hud_altitude columns:", [cols[i] for i in hud])


def group_drop(model, X, y, idx, rng):
    """AP drop when the columns in idx are shuffled together (same row order for all of them)."""
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    drops = []
    for _ in range(N_REPEATS):
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, idx] = X[perm][:, idx]
        drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
    return float(np.mean(drops))


per_fold, groups = [], []
for f in folds:
    train = dev[dev["role"] != f]
    test = dev[dev["role"] == f]
    Xtr, ytr = train[cols].to_numpy(float), train["label"].to_numpy(int)
    Xte, yte = test[cols].to_numpy(float), test["label"].to_numpy(int)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced_subsample",
                                   random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    r = permutation_importance(model, Xte, yte, scoring="average_precision",
                               n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1)
    per_fold.append(pd.DataFrame({"feature": cols, "fold": f, "drop": r.importances_mean}))
    if alt and hud:
        rng = np.random.default_rng(SEED)
        groups.append(dict(fold=f,
                           alt_error=group_drop(model, Xte, yte, alt, rng),
                           hud_altitude=group_drop(model, Xte, yte, hud, rng),
                           both=group_drop(model, Xte, yte, alt + hud, rng)))
    print("fold", f, "done")

pf = pd.concat(per_fold)
os.makedirs("results", exist_ok=True)
pf.to_csv(os.path.join("results", "permutation_importance.csv"), index=False)
imp = pf.groupby("feature")["drop"].agg(["mean", "std"]).sort_values("mean", ascending=False)

print("\nTop 12 features: mean AP drop over 5 folds (std = spread between folds)")
print(imp.head(12).round(4).to_string())
print("\nFeatures with mean drop below 0.001:", int((imp["mean"] < 0.001).sum()), "of", len(imp))

if groups:
    g = pd.DataFrame(groups)[["alt_error", "hud_altitude", "both"]].mean()
    print("\nGroup test, mean AP drop over 5 folds:")
    print(g.round(4).to_string())
    print("sum of the two groups separately:", round(g["alt_error"] + g["hud_altitude"], 4))
