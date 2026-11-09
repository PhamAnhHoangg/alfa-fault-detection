import pandas as pd
from evalkit import load, feature_lists, fit_rule, apply_rule, rule_text, evaluate

pd.set_option("display.width", 250)
df = load()
sym, allf = feature_lists(df)
dev = df[df["role"] != "test"]          # locked test set is NOT used

COLS = ["fold", "rule", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min",
        "det_eng", "pre_eng", "det_oth", "pre_oth", "med_delay_s"]

def run(feats, k, title):
    rows = []
    for i in range(5):
        train = dev[dev["role"] != f"cv{i}"]
        val = dev[dev["role"] == f"cv{i}"]
        rule = fit_rule(train, feats)
        r = evaluate(val, apply_rule(val, rule), k_persist=k)
        r["fold"] = f"cv{i}"
        r["rule"] = rule_text(rule)
        rows.append(r)
    out = pd.DataFrame(rows)[COLS]
    print("\n=== " + title + " ===")
    print(out.round(2).to_string(index=False))
    print("mean F1: %.2f (std %.2f)" % (out["f1"].mean(), out["f1"].std()))

run(sym, 1, "A: symptom only, alarm on any single second")
run(sym, 3, "A: symptom only, alarm needs 3 seconds in a row")
run(allf, 3, "B: with cause signals, 3 seconds in a row")
