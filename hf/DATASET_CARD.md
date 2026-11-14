---
license: cc-by-4.0
task_categories:
- tabular-classification
tags:
- fault-detection
- uav
- time-series
size_categories:
- 1K<n<10K
---

# ALFA per-second fault detection table

One row per second per flight for 46 flights of the ALFA dataset (fixed-wing UAV with injected engine and control-surface faults), built by the project [alfa-fault-detection](https://github.com/PhamAnhHoangg/alfa-fault-detection).

## Files

- `table.csv`: 4260 rows. Columns `flight`, `session`, `day`, `fault_type`, `t` (second since the first IMU reading), `t_rel_fault` (seconds relative to the fault start), `label` (1 when t + 1 > fault start) and 30 features: 5 s mean, 5 s standard deviation and 10 s change of 10 signals. Columns starting with `cause_` and `rc_out` come from the recorded commands, which already contain the injected fault; leave them out for a fair sensor-only detector.
- `split.csv`: one row per flight with its role, `cv0` to `cv4` (development folds) or `test` (9 locked flights). The split is by recording session.

## Changes from the original data

Readings were averaged per second, short gaps of up to 5 s were filled forward, rolling features were added, and rows with t below 10 were dropped. One flight without ground truth was left out. Features only use the current and earlier seconds.

## Credit

Derived from the ALFA dataset (CC BY 4.0): A. Keipour, M. Mousaei and S. Scherer, "ALFA: A dataset for UAV fault and anomaly detection", The International Journal of Robotics Research 40(2-3), 515-520, 2021, doi:10.1177/0278364920966642. Original data: https://doi.org/10.1184/R1/12707963
