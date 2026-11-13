"""Model comparison on the dev folds: random forest vs CatBoost vs a small Keras network.
Dev flights only. The locked test set is not used."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"
import time
import numpy as np
import pandas as pd
import keras
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import evalkit

SEEDS = [0, 1, 2, 3, 4]
THRESHOLD = 0.5
K_PERSIST = 3

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"].reset_index(drop=True)
sym, feats = evalkit.feature_lists(dev)
cols = [c for c in sym if not c.startswith("rc_out")]
folds = sorted(dev["role"].unique())
print("dev rows:", len(dev), "| features:", len(cols), "| folds:", folds)
print("NaN values in features:", int(dev[cols].isna().sum().sum()))


def fit_predict(name, train, test, seed):
    """Train on `train`, return the fault probability for every row of `test`."""
    Xtr = train[cols].to_numpy(float)
    ytr = train["label"].to_numpy(int)
    Xte = test[cols].to_numpy(float)

    if name == "forest":
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       class_weight="balanced_subsample",
                                       random_state=seed, n_jobs=-1)
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]

    if name == "catboost":
        model = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                   auto_class_weights="Balanced",
                                   random_seed=seed, verbose=0, allow_writing_files=False)
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]

    # small neural network: scale with the training rows only
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = np.nan_to_num(scaler.transform(Xtr))
    Xte_s = np.nan_to_num(scaler.transform(Xte))
    keras.utils.set_random_seed(seed)
    net = keras.Sequential([
        keras.layers.Input(shape=(Xtr_s.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    net.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    n1 = int(ytr.sum())
    n0 = len(ytr) - n1
    weights = {0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)}
    net.fit(Xtr_s, ytr, epochs=60, batch_size=64, class_weight=weights, verbose=0)
    proba = net.predict(Xte_s, verbose=0).ravel()
    keras.backend.clear_session()
    return proba


rows = []
saved = []
for name in ["forest", "catboost", "mlp"]:
    t0 = time.time()
    for seed in SEEDS:
        for f in folds:
            train = dev[dev["role"] != f]
            test = dev[dev["role"] == f]
            proba = fit_predict(name, train, test, seed)
            res = evalkit.evaluate(test, proba >= THRESHOLD, k_persist=K_PERSIST)
            res["ap"] = average_precision_score(test["label"], proba)
            res.update(model=name, seed=seed, fold=f)
            keep = test[[c for c in ["flight", "role", "fault_type", "t", "t_rel_fault", "label"] if c in test.columns]].copy()
            keep["proba"] = proba
            keep["model"] = name
            keep["seed"] = seed
            saved.append(keep)
            rows.append(res)
    print(f"{name} finished in {time.time() - t0:.0f} s")

os.makedirs("results", exist_ok=True)
pd.concat(saved).to_csv(os.path.join("results", "oof_models.csv"), index=False)
out = pd.DataFrame(rows)
out.to_csv(os.path.join("results", "model_comparison.csv"), index=False)

metrics = ["f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min", "med_delay_s", "ap"]
per_seed = out.groupby(["model", "seed"])[metrics].mean()
g = per_seed.groupby("model")
table = g[metrics].mean().round(2)
table["f1_std_over_seeds"] = g["f1"].std().round(3)
print("\nMean over the 5 folds, then mean over the 5 seeds:")
print(table.to_string())

print("\nSanity check: forest, seed 0, mean fold F1 =",
      round(out[(out.model == "forest") & (out.seed == 0)]["f1"].mean(), 2),
      "(earlier result: 0.51)")

print("\nF1 per fold, seed 0:")
print(out[out.seed == 0].pivot(index="fold", columns="model", values="f1").round(2).to_string())
