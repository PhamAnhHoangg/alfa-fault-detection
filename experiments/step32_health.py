import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score as AP, brier_score_loss
from evalkit import load, feature_lists, fit_rule, apply_rule, persist, evaluate

pd.set_option("display.width", 250)
df = load()
sym, _ = feature_lists(df)
sensors = [c for c in sym if not c.startswith("rc_out")]
dev = df[df["role"] != "test"].copy().reset_index(drop=True)      # locked test set is NOT used
TEST_F1_F = 0.50          # forest F1 on the locked test set, from Step 22 (not recomputed here)
y = dev["label"].to_numpy() == 1
base = y.mean()

def forest(weighted=True, n=300):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=n, min_samples_leaf=5,
                                                class_weight="balanced_subsample" if weighted else None,
                                                random_state=0, n_jobs=-1))

folds = [(dev["role"] != f"cv{i}", dev["role"] == f"cv{i}") for i in range(5)]
oof = pd.Series(np.nan, index=dev.index)
oof_nw = pd.Series(np.nan, index=dev.index)
dev["predF"] = False
dev["predA"] = False
tr_ap, va_ap, f1s = [], [], []
for mtr, mva in folds:
    tr, va = dev[mtr], dev[mva]
    m = forest().fit(tr[sensors], tr["label"])
    p_tr, p_va = m.predict_proba(tr[sensors])[:, 1], m.predict_proba(va[sensors])[:, 1]
    oof.loc[mva] = p_va
    tr_ap.append(AP(tr["label"], p_tr)); va_ap.append(AP(va["label"], p_va))
    f1s.append(evaluate(va, p_va >= 0.5, k_persist=3)["f1"])
    dev.loc[mva, "predF"] = persist(va, p_va >= 0.5, 3)
    m2 = forest(weighted=False).fit(tr[sensors], tr["label"])
    oof_nw.loc[mva] = m2.predict_proba(va[sensors])[:, 1]
    dev.loc[mva, "predA"] = persist(va, apply_rule(va, fit_rule(tr, sensors)), 3)

def prf(pred):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

cards = []

# ---------- precision / recall
print("=== PRECISION / RECALL (pooled development folds, 3 s in a row) ===")
rec_min = 1.0
for name, col in (("F forest", "predF"), ("A rule", "predA")):
    p, r, f = prf(dev[col].to_numpy())
    print(f"{name}: precision {p:.2f} | recall {r:.2f} | F1 {f:.2f}")
    per = {}
    for ft in ("engine", "aileron", "rudder", "elevator"):
        mk = ((dev["fault_type"] == ft) & y).to_numpy()
        per[ft] = dev.loc[mk, col].mean()
    print("   recall by fault type:", {k: round(v, 2) for k, v in per.items()})
    if name == "F forest":
        rec_min = min(per.values()); prec_f = p
st = "WARN" if (rec_min < 0.30 or prec_f < 0.5) else "OK"
cards.append(("Precision / recall", st, f"F precision {prec_f:.2f}; worst fault-type recall {rec_min:.2f}"))

# ---------- class imbalance
print("\n=== CLASS IMBALANCE ===")
ap_w, ap_nw = AP(y, oof), AP(y, oof_nw)
lift = ap_w / base
shares = dev.loc[y, "fault_type"].value_counts(normalize=True)
fold_share = [dev.loc[mva, "label"].mean() for _, mva in folds]
print(f"fault share of rows: {base:.3f} (per fold {min(fold_share):.2f} to {max(fold_share):.2f})")
print(f"pooled AP with class weights: {ap_w:.2f} | without: {ap_nw:.2f} | lift over chance: {lift:.1f}x")
print("share of fault rows by type:", shares.round(2).to_dict())
st = "WARN" if (lift < 2 or shares.max() > 0.5) else "OK"
cards.append(("Class imbalance", st, f"fault share {base:.0%}; AP lift {lift:.1f}x; largest fault type {shares.idxmax()} {shares.max():.0%} of fault rows"))

# ---------- calibration
print("\n=== CALIBRATION (out-of-fold forest probabilities) ===")
p = oof.to_numpy()
idx = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
ece, rows = 0.0, []
for b in range(10):
    mk = idx == b
    if mk.sum() == 0:
        continue
    ece += abs(p[mk].mean() - y[mk].mean()) * mk.sum() / len(p)
    rows.append({"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "rows": int(mk.sum()),
                 "mean_predicted": round(p[mk].mean(), 2), "observed_fault_rate": round(y[mk].mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))
brier = brier_score_loss(y, p); brier0 = base * (1 - base)
print(f"Brier {brier:.3f} (always predicting the base rate: {brier0:.3f}) | skill {1 - brier / brier0:.2f} | ECE {ece:.3f}")
st = "OK" if ece < 0.05 else ("WARN" if ece < 0.15 else "FAIL")
cards.append(("Calibration", st, f"ECE {ece:.3f}; Brier skill {1 - brier / brier0:.2f}"))

# ---------- overfitting
print("\n=== OVERFITTING ===")
gap_tv = np.mean(tr_ap) - np.mean(va_ap)
dev_f1 = float(np.mean(f1s)); gap_dt = dev_f1 - TEST_F1_F
print(f"forest AP on its own training rows: {np.mean(tr_ap):.2f} | on held-out sessions: {np.mean(va_ap):.2f} | gap {gap_tv:.2f} (a forest always fits its training rows)")
print(f"dev F1 {dev_f1:.2f} vs locked-test F1 {TEST_F1_F:.2f} | gap {gap_dt:+.2f}")
st = "OK" if abs(gap_dt) <= 0.05 else ("WARN" if abs(gap_dt) <= 0.10 else "FAIL")
cards.append(("Overfitting", st, f"dev-to-test F1 gap {gap_dt:+.2f}; train-to-validation AP gap {gap_tv:.2f}"))

# ---------- bias / variance
print("\n=== BIAS / VARIANCE (learning curve: share of training flights used) ===")
curve = []
for frac in (0.25, 0.5, 0.75, 1.0):
    a_tr, a_va = [], []
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        for mtr, mva in folds:
            tr, va = dev[mtr], dev[mva]
            fl = tr["flight"].unique()
            keep = rng.choice(fl, size=max(2, int(round(frac * len(fl)))), replace=False)
            sub = tr[tr["flight"].isin(keep)]
            if sub["label"].sum() == 0:
                continue
            m = forest(n=150).fit(sub[sensors], sub["label"])
            a_va.append(AP(va["label"], m.predict_proba(va[sensors])[:, 1]))
            a_tr.append(AP(sub["label"], m.predict_proba(sub[sensors])[:, 1]))
    curve.append({"train_flights_share": frac, "train_AP": np.mean(a_tr), "validation_AP": np.mean(a_va)})
cv = pd.DataFrame(curve)
print(cv.round(2).to_string(index=False))
print(f"fold-to-fold F1: mean {dev_f1:.2f}, std {np.std(f1s, ddof=1):.2f}")
gain = cv.loc[3, "validation_AP"] - cv.loc[1, "validation_AP"]
verdict = "variance-limited: more flights still help" if gain > 0.03 else "flat curve: more of the same data will not help much"
cards.append(("Bias / variance", "INFO", f"AP gain from 50% to 100% of flights {gain:+.2f} -> {verdict}; fold std {np.std(f1s, ddof=1):.2f}"))

# ---------- drift
print("\n=== DRIFT ACROSS RECORDING DAYS (July to October 2018) ===")
days = sorted(dev["day"].unique())
fw = []
for k in range(1, len(days)):
    tr, va = dev[dev["day"].isin(days[:k])], dev[dev["day"] == days[k]]
    if tr["label"].sum() == 0 or va["label"].sum() == 0:
        print(f"train on days before {days[k]} -> {days[k]}: skipped (no fault rows on one side)")
        continue
    m = forest(n=150).fit(tr[sensors], tr["label"])
    a = AP(va["label"], m.predict_proba(va[sensors])[:, 1])
    fw.append(a)
    print(f"train on earlier days ({len(tr)} rows) -> {days[k]}: AP {a:.2f} (chance {va['label'].mean():.2f})")
m_all = forest().fit(dev[sensors], dev["label"])
top = pd.Series(m_all[-1].feature_importances_, index=sensors).sort_values(ascending=False).head(6).index
def psi(a, b, bins=10):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    e[0], e[-1] = -np.inf, np.inf
    pa = np.clip(np.histogram(a, e)[0] / len(a), 1e-4, None)
    pb = np.clip(np.histogram(b, e)[0] / len(b), 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))
early = dev[dev["day"].isin(days[:2]) & (dev["label"] == 0)]
late = dev[~dev["day"].isin(days[:2]) & (dev["label"] == 0)]
psis = {c: round(psi(early[c].to_numpy(), late[c].to_numpy()), 2) for c in top}
print("feature shift, normal rows, first 2 days vs later days (PSI; above 0.25 = large):", psis)
fw_mean = float(np.mean(fw)) if fw else np.nan
st = "OK" if (fw and fw_mean >= np.mean(va_ap) - 0.10 and max(psis.values()) < 0.25) else "WARN"
cards.append(("Model drift", st, f"forward-in-time AP {fw_mean:.2f} vs session-CV AP {np.mean(va_ap):.2f}; max PSI {max(psis.values()):.2f}"))

# ---------- leakage tests
print("\n=== DATA LEAKAGE TESTS ===")
bad = [c for c in sensors if any(w in c.lower() for w in ("fail", "label", "t_rel", "fault"))]
print("label-like feature names:", bad if bad else "none")
sh_ap = []
for mtr, mva in folds:
    tr, va = dev[mtr], dev[mva]
    yy = tr["label"].to_numpy().copy(); np.random.default_rng(0).shuffle(yy)
    sh_ap.append(AP(va["label"], forest(n=150).fit(tr[sensors], yy).predict_proba(va[sensors])[:, 1]))
print(f"shuffled-label AP: {np.mean(sh_ap):.2f} (fault share {base:.2f}; real AP {np.mean(va_ap):.2f})")
g = dev.groupby("flight")
ok = ((dev["t"] - g["t"].shift(1)) == 1).to_numpy()
lag = g[sensors].shift(1)
lag.loc[~ok, :] = np.nan
dl = dev.copy(); dl[sensors] = lag; dl = dl.dropna(subset=sensors)
l_f1, l_dl, r_dl = [], [], []
for i in range(5):
    tr, va = dl[dl["role"] != f"cv{i}"], dl[dl["role"] == f"cv{i}"]
    pv = forest(n=150).fit(tr[sensors], tr["label"]).predict_proba(va[sensors])[:, 1]
    r = evaluate(va, pv >= 0.5, k_persist=3); l_f1.append(r["f1"]); l_dl.append(r["med_delay_s"])
for mtr, mva in folds:
    pass
r_dl = [evaluate(dev[mva], dev.loc[mva, "predF"].to_numpy(), k_persist=1)["med_delay_s"] for _, mva in folds]
print(f"1 s lag: F1 {np.mean(l_f1):.2f} (as built {dev_f1:.2f}) | median delay {pd.Series(l_dl).mean():.1f} s (as built {pd.Series(r_dl).mean():.1f} s)")
d_f1 = []
for d in days:
    tr, va = dev[dev["day"] != d], dev[dev["day"] == d]
    if va["label"].sum() == 0 or tr["label"].sum() == 0:
        continue
    pv = forest(n=150).fit(tr[sensors], tr["label"]).predict_proba(va[sensors])[:, 1]
    r = evaluate(va, pv >= 0.5, k_persist=3); d_f1.append(r["f1"])
    print(f"   leave day {d} out: F1 {r['f1']:.2f}")
dayout = float(np.mean(d_f1))
st = "FAIL" if (bad or np.mean(sh_ap) > base + 0.10) else ("WARN" if (np.mean(l_f1) < dev_f1 - 0.08 or dayout < dev_f1 - 0.15) else "OK")
cards.append(("Data leakage", st, f"shuffled AP {np.mean(sh_ap):.2f} vs share {base:.2f}; 1 s lag F1 {np.mean(l_f1):.2f}; leave-day-out F1 {dayout:.2f}"))

print("\n================ SCORECARD (rules of thumb) ================")
for n, s, d in cards:
    print(f"{n:<20} {s:<5} {d}")
print("\nNot testable by this script: signal screening saw the test flights; 1 flight per non-engine type in the test set;")
print("the command-side detector D2 is excluded (it reads the injected command).")
