import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
table = pd.read_csv(r"E:\ml-course\data\table.csv")
split = pd.read_csv(r"E:\ml-course\data\split.csv")[["flight", "role"]]
df = table.merge(split, on="flight")
before = len(df)
df = df[~(df["t_rel_fault"] > 30)]      # cap: first 30 s after the fault only
print("rows after 30 s cap:", len(df), "(dropped", before - len(df), ")")
dev = df[df["role"] != "test"].copy()   # the locked test set is NOT used here
print("development rows:", len(dev), " label counts:", dev["label"].value_counts().to_dict())

FEATS = [c for c in df.columns if "__" in c]
SYM = [c for c in FEATS if not c.startswith("cause_")]

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

def evaluate(d, pred):
    d = d.assign(pred=pred)
    y = d["label"].to_numpy() == 1
    p = d["pred"].to_numpy()
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fa = fp / max((~y).sum() / 60, 1e-9)
    res = dict(f1=f1, prec=prec, rec=rec, fa_s_per_min=fa)
    delays = []
    for grp, key in (("engine", "eng_det"), ("other", "oth_det")):
        sub = d[(d["fault_type"] == "engine") if grp == "engine"
                else (~d["fault_type"].isin(["engine", "none"]))]
        hits, total = 0, 0
        for name, g in sub.groupby("flight"):
            total += 1
            h = g[(g["label"] == 1) & g["pred"]]
            if len(h):
                hits += 1
                delays.append(max(h["t_rel_fault"].min(), 0))
        res[key] = f"{hits}/{total}"
    res["med_delay_s"] = float(np.median(delays)) if delays else np.nan
    return res

def run(feats, title):
    rows = []
    for k in range(5):
        train = dev[dev["role"] != f"cv{k}"]
        val = dev[dev["role"] == f"cv{k}"]
        rule = fit_rule(train, feats)
        r = evaluate(val, apply_rule(val, rule))
        r["fold"] = f"cv{k}"
        r["rule"] = f"{rule[1]} {'>=' if rule[2] == 'ge' else '<='} {rule[3]:.3g}"
        rows.append(r)
    out = pd.DataFrame(rows)[["fold", "rule", "f1", "prec", "rec", "fa_s_per_min", "eng_det", "oth_det", "med_delay_s"]]
    print("\n=== " + title + " ===")
    print(out.round(2).to_string(index=False))
    print("mean F1: %.2f  (std across folds %.2f)" % (out["f1"].mean(), out["f1"].std()))

print("\nreference: 'always normal' gets F1 = 0.00 (it never raises an alarm)")
run(SYM, "BASELINE A: symptom signals only (honest)")
run(FEATS, "BASELINE B: with the cause signals (throttle) allowed")
