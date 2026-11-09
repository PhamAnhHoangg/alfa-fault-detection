import pandas as pd
import evalkit
from evalkit import load, fit_rule, apply_rule, rule_text, persist, evaluate

evalkit.TABLE = r"E:\ml-course\data\table2.csv"
pd.set_option("display.width", 250)
df = load()
dev = df[df["role"] != "test"]          # locked test set is NOT used

COLS = ["fold", "rule", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min",
        "det_eng", "pre_eng", "det_oth", "pre_oth", "med_delay_s"]
res1, res2, iso = [], [], []
for i in range(5):
    tr = dev[dev["role"] != f"cv{i}"]
    va = dev[dev["role"] == f"cv{i}"]
    frozen = va["cmd_frozen_any"].to_numpy() == 1

    r1 = evaluate(va, frozen)
    r1["fold"], r1["rule"] = f"cv{i}", "frozen any"
    res1.append(r1)

    rule = fit_rule(tr, ["cause_rc_out_2__mean5"])
    thr = persist(va, apply_rule(va, rule), 3)
    pred = frozen | thr
    r2 = evaluate(va, pred)
    r2["fold"], r2["rule"] = f"cv{i}", "frozen OR " + rule_text(rule)
    res2.append(r2)

    hat = pd.Series("none", index=va.index)
    hat[thr] = "engine"
    hat[va["cmd_frozen_ch1"].to_numpy() == 1] = "elevator"
    hat[va["cmd_frozen_ch3"].to_numpy() == 1] = "rudder"
    hat[(va["cmd_frozen_ch4"] + va["cmd_frozen_ch5"]).to_numpy() > 0] = "aileron"
    m = (va["label"].to_numpy() == 1) & pred
    iso.append(pd.DataFrame({"true": va["fault_type"].to_numpy()[m], "guess": hat.to_numpy()[m]}))

for title, res in (("D1: frozen channel only", res1), ("D2: frozen channel OR throttle level", res2)):
    out = pd.DataFrame(res)[COLS]
    print("\n=== " + title + " ===")
    print(out.round(2).to_string(index=False))
    print("mean F1: %.2f (std %.2f)" % (out["f1"].mean(), out["f1"].std()))

iso = pd.concat(iso)
print("\n=== D2 isolation: true fault type (rows) vs guessed surface (columns), flagged fault seconds, all folds ===")
print(pd.crosstab(iso["true"], iso["guess"]).to_string())
