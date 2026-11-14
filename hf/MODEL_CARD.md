---
license: cc-by-4.0
library_name: sklearn
tags:
- tabular-classification
- fault-detection
- uav
- time-series
datasets:
- {USER}/alfa-per-second-table
---

# ALFA fault detection: sensor random forest

A random forest that flags, second by second, whether a fixed-wing UAV is flying with a fault, using only past sensor data. It is the sensor-only detector F from the project [alfa-fault-detection](https://github.com/PhamAnhHoangg/alfa-fault-detection).

## What it reads

21 per-second features: the 5 s mean, 5 s standard deviation and 10 s change of alt_error, xtrack_error, hud_altitude, vel_lin_z, roll_gap, pitch_gap and yaw_gap. The exact order is stored in the bundle as `features`.

## Alarm rule

Raw alarm when the fault probability is at least 0.5. Final alarm when the raw alarm is on for 3 seconds in a row inside one flight. Both values are stored in the bundle (`threshold`, `k_persist`).

## How it was trained

300 trees, `min_samples_leaf=5`, `class_weight="balanced_subsample"`, `random_state=0`, scikit-learn 1.9.1. Trained on all 37 development flights (3279 seconds after the 30 s cap). The 9 locked test flights were not used.

## How good it is

With the same settings, the forest scored F1 0.50 on the locked test set and 0.51 on average over 5 development folds (fold range 0.35 to 0.78). It finds engine failures reliably. Aileron, rudder and elevator faults are caught only 12 to 27 percent of the time. The test set is 9 flights, so these numbers are rough.

## Use

```python
import joblib
b = joblib.load("forest.joblib")
proba = b["model"].predict_proba(X[b["features"]])[:, 1]
```

Load it with scikit-learn 1.9.1; other versions may warn or fail.

## Limits

One aircraft type, faults injected on purpose, a small test set. Not a safety system.

## Data credit

Trained on features built from the ALFA dataset (CC BY 4.0): A. Keipour, M. Mousaei and S. Scherer, "ALFA: A dataset for UAV fault and anomaly detection", The International Journal of Robotics Research 40(2-3), 515-520, 2021, doi:10.1177/0278364920966642. Data: https://doi.org/10.1184/R1/12707963
