"""Self-check 1: what does a model that ignores the sensors score? (dev folds only)

'prior' always outputs the fault rate of the training folds -> its average precision equals the
share of fault seconds in the test fold (the chance level).
'stratified' guesses fault / normal at random with the training fault rate, 20 random draws."""
import os
import numpy as np
import pandas as pd
import evalkit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score

K_PERSIST = 3
N_DRAWS = 20

oof = pd.read_csv(os.path.join("results", "oof_models.csv"))
rows_src = oof[(oof["model"] == "forest") & (oof["seed"] == 0)]   # row identity only, features unused
folds = sorted(rows_src["role"].unique())

prior_ap, strat = [], []
for f in folds:
    test = rows_src[rows_src["role"] == f].reset_index(drop=True)
    train = rows_src[rows_src["role"] != f]
    ytr, yte = train["label"].to_numpy(), test["label"].to_numpy()
    Xtr, Xte = np.zeros((len(ytr), 1)), np.zeros((len(yte), 1))

    prior = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    prior_ap.append(dict(fold=f, fault_share=yte.mean(),
                         ap=average_precision_score(yte, prior.predict_proba(Xte)[:, 1])))

    for s in range(N_DRAWS):
        d = DummyClassifier(strategy="stratified", random_state=s).fit(Xtr, ytr)
        raw = d.predict(Xte).astype(bool)
        res = evalkit.evaluate(test, raw, k_persist=K_PERSIST)
        res.update(fold=f, draw=s)
        strat.append(res)

prior_ap = pd.DataFrame(prior_ap)
strat = pd.DataFrame(strat)

print("Chance level (DummyClassifier 'prior'): average precision = share of fault seconds")
print(prior_ap.round(3).to_string(index=False))
print("mean chance AP:", round(prior_ap["ap"].mean(), 3))

print("\nRandom guessing ('stratified', alarm rule 3 in a row), mean over 5 folds x 20 draws:")
cols = ["f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min"]
print(strat[cols].mean().round(3).to_string())

print("\nSame measures for the real models (fixed 0.5, from the model comparison):")
mc = pd.read_csv(os.path.join("results", "model_comparison.csv"))
print(mc.groupby("model")[cols + ["ap"]].mean().round(3).to_string())
