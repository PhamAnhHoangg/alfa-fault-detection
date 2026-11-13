"""Self-check 4: does the forest improve with more training flights? (dev folds only)

Like sklearn's learning_curve, but the training set grows by whole FLIGHTS, not rows, because
rows from one flight are near-copies of each other."""
import os, config
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import evalkit

FRACS = [0.25, 0.5, 0.75, 1.0]
DRAWS = 3
THRESHOLD = 0.5
K_PERSIST = 3

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"].reset_index(drop=True)
sym, feats = evalkit.feature_lists(dev)
cols = [c for c in sym if not c.startswith("rc_out")]
folds = sorted(dev["role"].unique())
print("dev rows:", len(dev), "| features:", len(cols), "| dev flights:", dev["flight"].nunique())

rows = []
for frac in FRACS:
    for f in folds:
        train_all = dev[dev["role"] != f]
        test = dev[dev["role"] == f].reset_index(drop=True)
        flights = np.array(sorted(train_all["flight"].unique()))
        Xte, yte = test[cols].to_numpy(float), test["label"].to_numpy(int)
        for draw in range(DRAWS):
            rng = np.random.default_rng(1000 * draw + int(frac * 100))
            n = max(3, int(round(frac * len(flights))))
            for _ in range(20):                     # redraw if the subset has only one class
                chosen = rng.choice(flights, size=n, replace=False)
                tr = train_all[train_all["flight"].isin(chosen)]
                ytr = tr["label"].to_numpy(int)
                if 0 < ytr.sum() < len(ytr):
                    break
            else:
                raise SystemExit(f"no subset with both classes after 20 draws (fold {f}, frac {frac})")
            model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                           class_weight="balanced_subsample",
                                           random_state=draw, n_jobs=-1)
            model.fit(tr[cols].to_numpy(float), ytr)
            proba = model.predict_proba(Xte)[:, 1]
            res = evalkit.evaluate(test, proba >= THRESHOLD, k_persist=K_PERSIST)
            rows.append(dict(frac=frac, fold=f, draw=draw, n_flights=n,
                             ap=average_precision_score(yte, proba), f1=res["f1"]))
    print("training share", frac, "done")

out = pd.DataFrame(rows)
os.makedirs(config.PROJECT / "results", exist_ok=True)
out.to_csv(os.path.join(config.PROJECT, "results", "learning_curve.csv"), index=False)

per_fold = out.groupby(["frac", "fold"])[["n_flights", "ap", "f1"]].mean()
table = per_fold.groupby("frac").agg(n_flights=("n_flights", "mean"),
                                     ap=("ap", "mean"), ap_fold_std=("ap", "std"),
                                     f1=("f1", "mean"), f1_fold_std=("f1", "std"))
print("\nMean over 5 held-out folds (and 3 random flight subsets per fold):")
print(table.round(3).to_string())
