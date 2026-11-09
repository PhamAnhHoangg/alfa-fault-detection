import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from evalkit import load, feature_lists, evaluate

pd.set_option("display.width", 250)
df = load()
sym, allf = feature_lists(df)
dev = df[df["role"] != "test"]          # locked test set is NOT used

COLS = ["fold", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min",
        "det_eng", "pre_eng", "det_oth", "pre_oth", "med_delay_s", "ap"]

def run(q, k=3):
    rows = []
    for i in range(5):
        tr = dev[dev["role"] != f"cv{i}"]
        va = dev[dev["role"] == f"cv{i}"]
        normal = tr[tr["label"] == 0]                      # fault rows never used for fitting
        prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(normal[sym])
        iso = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1)
        iso.fit(prep.transform(normal[sym]))
        thr = pd.Series(-iso.score_samples(prep.transform(normal[sym]))).quantile(q)
        score = -iso.score_samples(prep.transform(va[sym]))
        r = evaluate(va, score > thr, k_persist=k)
        r["ap"] = average_precision_score(va["label"], score)
        r["fold"] = f"cv{i}"
        rows.append(r)
    out = pd.DataFrame(rows)[COLS]
    print(f"\n=== isolation forest, threshold = {q:.0%} of normal training scores, 3 in a row ===")
    print(out.round(2).to_string(index=False))
    print("mean F1: %.2f (std %.2f) | mean AP: %.2f" % (out["f1"].mean(), out["f1"].std(), out["ap"].mean()))

run(0.98)
run(0.99)
