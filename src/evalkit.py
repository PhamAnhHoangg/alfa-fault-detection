import config
import numpy as np
import pandas as pd

TABLE = str(config.DATA / r"table.csv")
SPLIT = str(config.DATA / r"split.csv")

def load(cap=30):
    table = pd.read_csv(TABLE)
    split = pd.read_csv(SPLIT)[["flight", "role"]]
    df = table.merge(split, on="flight").sort_values(["flight", "t"]).reset_index(drop=True)
    return df[~(df["t_rel_fault"] > cap)].reset_index(drop=True)

def feature_lists(df):
    feats = [c for c in df.columns if "__" in c]
    sym = [c for c in feats if not c.startswith("cause_")]
    return sym, feats

def f1_of(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def fit_rule(train, feats):
    y = train["label"].to_numpy() == 1
    best = (-1, None, None, None)
    for f in feats:
        x = train[f].to_numpy()
        for thr in np.unique(np.quantile(x, np.linspace(0.02, 0.98, 49))):
            for d in ("ge", "le"):
                pred = (x >= thr) if d == "ge" else (x <= thr)
                s = f1_of(pred, y)
                if s > best[0]:
                    best = (s, f, d, thr)
    return best

def apply_rule(d, rule):
    _, f, direction, thr = rule
    x = d[f].to_numpy()
    return (x >= thr) if direction == "ge" else (x <= thr)

def rule_text(rule):
    return f"{rule[1]} {'>=' if rule[2] == 'ge' else '<='} {rule[3]:.3g}"

def persist(d, pred, k):
    s = pd.Series(np.asarray(pred, dtype=int), index=d.index)
    r = s.groupby(d["flight"]).transform(lambda x: x.rolling(k, min_periods=k).min())
    return r.fillna(0).astype(bool).to_numpy()

def evaluate(d, pred, k_persist=1, pre_window=20):
    d = d.copy()
    pred = np.asarray(pred, dtype=bool)
    if k_persist > 1:
        pred = persist(d, pred, k_persist)
    d["pred"] = pred
    y = d["label"].to_numpy() == 1
    p = d["pred"].to_numpy()
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    eng = (d["fault_type"] == "engine").to_numpy()
    oth = (~d["fault_type"].isin(["engine", "none"])).to_numpy()
    res = dict(f1=f1, prec=prec,
               rec_eng=float(p[y & eng].mean()) if (y & eng).any() else np.nan,
               rec_oth=float(p[y & oth].mean()) if (y & oth).any() else np.nan,
               fa_s_per_min=fp / max((~y).sum() / 60, 1e-9))
    delays = []
    for name, mask in (("eng", eng), ("oth", oth)):
        det = pre = n = 0
        for _, g in d[mask].groupby("flight"):
            n += 1
            hit = g[(g["label"] == 1) & g["pred"]]
            if len(hit):
                det += 1
                delays.append(max(hit["t_rel_fault"].min(), 0))
            pre += int(g[(g["label"] == 0) & (g["t_rel_fault"] >= -pre_window)]["pred"].any())
        res["det_" + name] = f"{det}/{n}"
        res["pre_" + name] = f"{pre}/{n}"
    res["med_delay_s"] = float(np.median(delays)) if delays else np.nan
    return res

