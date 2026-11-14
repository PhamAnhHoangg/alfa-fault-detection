# Fault detection on real UAV flights (ALFA dataset): sensors vs. command channels

[![tests](https://github.com/PhamAnhHoangg/alfa-fault-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/PhamAnhHoangg/alfa-fault-detection/actions/workflows/tests.yml)

Detects the start of injected faults (engine failure and control-surface faults) in real flights of a fixed-wing UAV,
one second at a time, using only data from the past. Three detectors are compared on a locked test set, and the
project documents a data leak that makes the task look much easier than it is.

## Summary

- **Sensor-only detectors** (a one-feature rule and a random forest) reliably detect **engine** faults (5/5 test flights,
  median delay about 5 s) but are weak on aileron, rudder and elevator faults.
- **A command-side detector** (a frozen control channel, or a throttle cut) catches **every** fault type
  (8/8 test fault flights, F1 0.87, median delay about 3 s) and also says which surface failed.
- **Why that is not a fair win:** in ALFA the recorded servo commands (`rc-out`) already contain the injected fault.
  A frozen channel *is* the injection. This detector is therefore a ceiling that shows how easy the task becomes when
  the injection is visible, not a detector that would work on a real aircraft. The sensor-only result is the honest one.

## How to read this repo

1. [Summary](#summary) and [Results](#results) give the main finding in two minutes.
2. [Limitations](#limitations) says what the numbers do not show.
3. [Visual guide](#visual-guide) walks through the figures as GIFs.
4. [docs/ALFA_fault_detection_documentation.md](docs/ALFA_fault_detection_documentation.md) explains every step, number and formula.
5. [docs/code_index.md](docs/code_index.md) points from each idea to the script and line numbers that compute it.
6. [How to run](#how-to-run) reproduces everything from the raw data.

## Data

[ALFA: A dataset for UAV fault and anomaly detection](https://doi.org/10.1184/R1/12707963) (CMU AirLab, CC BY 4.0).
47 processed flights of a fixed-wing UAV with sudden engine failures and control-surface faults, with ground-truth
fault times. One flight has no ground truth and is excluded, leaving 46. The data is **not** included in this repository.

Fault onset = first timestamp of the `failure_status-*` topics. Those topics are used only to build the label and are never
a model input.

## Evaluation design

- **One row per second**, built from trailing windows only (no future data): 5 s mean, 5 s standard deviation and
  10 s change for each signal. Sensor signals: altitude error, cross-track error, altitude, vertical velocity, and
  roll / pitch / yaw gap (measured minus commanded, angles wrapped to +/-180 degrees).
- **Splits by recording session**, so flights from the same session never end up on both sides. 9 flights were locked
  as a **test set** before any model was trained (5 engine, aileron, rudder, elevator, 1 normal). The other 37 flights
  were used for 5-fold session-grouped cross-validation.
- Only the **first 30 s after the fault** are scored as fault. Accuracy is not used (always predicting "normal" is right
  about 81% of the time). Reported instead: precision, recall, F1, false-alarm seconds per minute, flights detected,
  detection delay, and a **pre-fault control** (alarms in the 20 s before the fault, which show the chance level).
- An alarm needs **3 seconds in a row** (the frozen-channel feature already contains a 3 s window).

## Detectors

| Name | Inputs | Rule |
|---|---|---|
| **A** sensor rule | sensors only | one threshold on the 5 s mean vertical velocity, fitted on development data |
| **F** sensor forest | 21 sensor features | random forest, class-balanced, threshold 0.5 |
| **D2** command-side | `rc-out` channels | any of channels 1, 3, 4, 5 frozen for 3 s, OR throttle output below a fitted level |

Channel meaning is **inferred** from the data and flight names (the dataset README does not document it):
channel 1 = elevator, 2 = throttle, 3 = rudder, 4 / 5 = left / right aileron.

## Results

Development = 5-fold cross-validation on 37 flights. Test = 9 locked flights, scored once.

| Detector | Dev F1 | Test F1 | Test precision | Engine flights alarmed | Other-fault flights alarmed | False alarms (s/min) | Median delay (s) |
|---|---|---|---|---|---|---|---|
| A sensor rule | 0.48 +/- 0.18 | 0.46 | 0.90 | 5/5 | 1/3 | 0.44 | 5.1 |
| F sensor forest | 0.51 +/- 0.16 | 0.50 | 1.00 | 5/5 | 2/3 | 0.00 | 5.2 |
| D2 command-side | 0.85 +/- 0.09 | 0.87 | 0.98 | 5/5 | 3/3 | 0.26 | 3.1 |

No detector raised an alarm in the 20 s before the fault on any test fault flight. Test F1 is close to development F1
for all three, so the estimates did not collapse on unseen flights. An anomaly detector (isolation forest trained on
normal data only) was also tried and was clearly worse (dev F1 0.12 to 0.26), so it is not part of the final comparison.

**Fault isolation (D2, test set):** 94 of 119 flagged fault seconds named the right surface. Aileron (28/28),
elevator (8/8) and engine (58/58) were all right. The rudder flight scored 0/25 because it is a dual-fault flight
(rudder + left aileron): both channels 3 and 4 froze, but the rule outputs a single label. The evidence was correct;
the one-label rule could not express two faults.

## What the data revealed

1. Faults show up as **frozen command channels**: the affected output holds a constant value (often neutral, 1500
   microseconds) from the fault onset. Mean-level tests missed this for aileron faults because a surface frozen at neutral
   looks normal on average. Spread (variance) tests found it.
2. Because the injection is visible in the commands, a detector that reads them is not a physical detector. Separating
   "sensor-side" from "command-side" detectors is the main contribution of this project.

| Step by step | Final figure |
|---|---|
| ![Engine fault, test flight (GIF)](figures/gif_fig1.gif) | ![Engine fault, test flight](figures/fig1_engine_flight.png) |
| Step by step | Final figure |
|---|---|
| ![Frozen command channels, dual-fault test flight (GIF)](figures/gif_fig2.gif) | ![Frozen command channels, dual-fault test flight](figures/fig2_frozen_channels.png) |

## Limitations

- Small data: 46 flights, **one flight per non-engine fault type in the test set**, and only about 21 s of normal flight
  in the test set, so false-alarm rates there are weakly measured. No confidence intervals.
- Some design choices (the 3 s freeze definition, the channel choice) were made after looking at the development flights,
  so development scores are slightly optimistic.
- D2 reads the injected command and would not transfer to a real aircraft. Channel meanings are inferred.
- One aircraft type and one dataset; no check on other platforms.
- The forest threshold (0.5) was not tuned.
- Recall on aileron, rudder and elevator faults is low on the dev folds (0.04, 0.02, 0.15). The forest mainly detects engine faults.
- Results vary by recording day (forward-in-time AP down to 0.27, PSI 1.46 on one feature), so they may not carry over to new days or aircraft.
- Reported detection delays are about 1 s optimistic (see the 1 s lag test above).

## Model health checks

Extra checks on the random forest, run on the **dev folds only** (the locked test set was not used). Script: `experiments/step32_health.py`.

| Check | Result | Reading |
|---|---|---|
| Precision / recall | 0.92 / 0.38 | Alarms are usually right, but most fault seconds are missed |
| Recall by fault type | engine 0.61, elevator 0.15, aileron 0.04, rudder 0.02 | Mostly an engine-fault detector |
| Class imbalance | 15% fault seconds; AP 0.65 vs 0.155 for random guessing (4.2x) | Real skill above chance. Class weights did not change AP (0.65 either way) |
| Calibration | Brier skill 0.35, ECE 0.061 | Overestimates between 0.1 and 0.5, roughly right or low above 0.5 |
| Overfitting | train AP 0.99, held-out AP 0.66 | Large gap, but dev F1 (0.51) and locked-test F1 (0.50) agree |
| Learning curve | validation AP 0.61, 0.64, 0.65, 0.65 | Flat: more of the same flights is unlikely to help much |
| Drift | forward-in-time AP by day 0.84, 0.27, 0.64, 0.82; PSI 1.46 (alt_error mean) and 0.33 (hud_altitude mean) | Performance depends on the recording day |
| Leakage | shuffled-label AP 0.18 (chance 0.155); no label-like feature names; 1 s lag: F1 0.51 to 0.47, median delay 5.0 to 6.0 s | No leak found. Reported delays are about 1 s optimistic |

Leave-one-day-out F1: 0.77, 0.77, 0.04, 0.50, 0.62. The day with AP 0.27 (2018-09-11) holds most of the non-engine faults.

### Figures

| Step by step | Final figure |
|---|---|
| ![Recall by fault type (GIF)](figures/gif_fig3.gif) | ![Recall by fault type](figures/fig3_recall_by_type.png) |

Recall by fault type on the dev folds. The forest finds most engine-fault seconds but very few aileron, rudder or elevator ones. The 3-in-a-row alarm rule removes false alarms but costs recall (rudder 0.18 to 0.02). Elevator has only 13 fault seconds and rudder 44, so those bars are noisy.

| Step by step | Final figure |
|---|---|
| ![Alarms per dev flight (GIF)](figures/gif_fig5.gif) | ![Alarms per dev flight](figures/fig5_alarm_timeline.png) |

Each row is one dev flight, aligned at the fault start. Every engine flight raised an alarm after the fault, roughly 3 to 7 seconds later in most flights. Most aileron and rudder flights raised none, and a few alarms fire before the fault (false alarms).

| Step by step | Final figure |
|---|---|
| ![Confusion matrix (GIF)](figures/gif_fig4.gif) | ![Confusion matrix](figures/fig4_confusion_matrix.png) |

Confusion matrix of the 3-in-a-row alarm, as a share of each actual class: 99% of normal seconds stay quiet (1% false alarms) and 38% of fault seconds are detected.

These figures are produced by `src/06_oof_predictions.py` (out-of-fold predictions, saved to `results/`, which is not committed) and `src/07_health_figures.py`.

## Model comparison

The random forest was compared with two other model families on the same dev folds: the same 21 sensor
features, the same alarm rule (probability >= 0.5, three alarms in a row) and 5 random seeds. The settings were
fixed in advance and not tuned. The locked test set was not used.

| Model | F1 | Precision | Engine recall | Other-fault recall | False-alarm s/min | Median delay | AP |
|---|---|---|---|---|---|---|---|
| Random forest | 0.51 | 0.90 | 0.62 | 0.12 | 0.40 | 5.1 s | 0.66 |
| CatBoost | 0.54 | 0.82 | 0.66 | 0.19 | 1.02 | 5.1 s | 0.67 |
| Keras network (32-16-1) | 0.53 | 0.63 | 0.74 | 0.27 | 3.39 | 4.5 s | 0.61 |

Mean over the 5 folds, then over the 5 seeds. AP = average precision, a score that does not depend on the alarm
threshold.

| Step by step | Final figure |
|---|---|
| ![Model comparison (GIF)](figures/gif_fig8.gif) | ![Model comparison](figures/fig8_model_comparison.png) |

- The three models are close. The F1 differences (0.51 to 0.54) are smaller than the spread between folds
  (0.35 to 0.78 for every model), so no model is clearly better.
- Forest and CatBoost rank the seconds about equally well (AP 0.66 and 0.67); the network is lower (0.61). Its F1
  comes from a different trade-off: it catches more faults but raises about 8 times more false alarms than the forest.
- Recall on aileron, rudder and elevator faults stays between 0.12 and 0.27 for all three models. This suggests the
  limit is the information in the features rather than the model type (not tested further).
- Limits: 5 folds and 37 flights, fixed hyperparameters, and one fixed threshold of 0.5, which lands at a
  different operating point for each model.

Run it in an environment built from `requirements-tf.txt`: `python src/09_model_comparison.py`, then
`python src/10_comparison_figure.py`.
## Choosing the alarm threshold

So far an alarm fired when the model's fault probability reached a fixed 0.5. That number is arbitrary, and the same 0.5 lands at very different operating points for different models. Instead, the threshold can be set on purpose: take the model's scores on normal seconds only, from the other four folds, and set the threshold at the top `alpha` share of those scores (`src/12_conformal_threshold.py` for the forest, `src/13_conformal_models.py` for all three models). A fold never sets its own threshold. The usual conformal correction makes this promise valid only if seconds are exchangeable, which seconds within a flight are not, so the promise is checked on the held-out fold instead of trusted.

Dev folds only, mean over 5 folds and 5 seeds, alarm rule 3 seconds in a row. FA = false alarms per minute.

| Model | 0.5: F1 | 0.5: FA/min | alpha 0.05: F1 | alpha 0.05: engine recall | alpha 0.05: other-fault recall | alpha 0.05: FA/min |
|---|---|---|---|---|---|---|
| Forest | 0.51 | 0.40 | 0.53 | 0.67 | 0.17 | 1.20 |
| CatBoost | 0.54 | 1.02 | 0.53 | 0.66 | 0.18 | 1.09 |
| Network (MLP) | 0.53 | 3.39 | 0.52 | 0.62 | 0.22 | 1.15 |

What the numbers show:

- **The promise holds on average, not per fold.** At alpha 0.05 the share of normal seconds that raised a raw alarm was 0.052 to 0.054 on average, but ranged from 0.01 to 0.13 between folds. Fold cv4 reached 0.10 to 0.13 for all three models, about 2 to 2.5 times the target. I did not find out why; flight-to-flight differences are a likely reason.
- **At the same false-alarm level the three models are close.** The F1 gaps (0.01 to 0.02) are smaller than the spread between folds (0.35 to 0.78), so no model is declared the winner.
- **The network's high recall at 0.5 came with many false alarms.** At 0.5 it caught 74% of engine faults but raised 3.4 false alarms a minute. At about 1.1 per minute its engine recall (0.62) is the lowest of the three, and its other-fault recall (0.22) stays the highest.
- **At the strictest level (alpha 0.01) CatBoost was ahead**: F1 0.45 against 0.40 for the forest and 0.33 for the network. This is one setting and was not tested further.
- **The forest's fixed 0.5 behaves like alpha 0.02**, so the original choice was reasonable.

The locked test set was not used for any of this.

## Self-checks on the forest

Four small checks on the random forest, using scikit-learn tools. All use the dev folds only; the locked test set was not touched. Scripts: `src/14_dummy_baseline.py`, `15_calibration.py`, `16_permutation_importance.py`, `17_learning_curve.py`.

- **Baseline.** A model that ignores the sensors has an average precision (AP) equal to the share of fault seconds, 0.157. The real models score 0.61 to 0.67, about 4 times chance. Random guessing with the 3-seconds-in-a-row alarm rule scores F1 0.006, which is a very weak baseline, because that rule almost never fires on random guesses. The more useful baseline is the single-feature rule A: on the locked test set its F1 was 0.46 against 0.50 for the forest.
- **Calibration** (do scores behave like probabilities?). Brier score, lower is better, where always predicting the fault share gives 0.131: forest 0.085, CatBoost 0.089, network 0.131. The forest and CatBoost are usable as rough probabilities. The network is no better than chance on this measure. All three models were trained with class weights, so that alone does not explain it, and I did not find the reason. The conformal threshold above uses only the ranking of scores, so it is not affected.
- **Permutation importance** (which features does the forest rely on?). `vel_lin_z__mean5` is the clear leader: shuffling it costs 0.072 AP. Every other feature costs 0.026 or less, and for most of them the spread between folds is as large as the value itself, so the order below the top is not reliable. 7 of 21 features cost less than 0.001. Shuffling the `alt_error` and `hud_altitude` features together costs 0.185 AP, against 0.052 for the two groups shuffled separately, which means these duplicated features share credit and the altitude information matters much more than any single feature suggests. Why altitude matters was not tested.
- **Learning curve by flights** (would more flights help?). Training on 25%, 50%, 75% and 100% of the training flights (7.6 to 29.6 flights on average) gave AP 0.61, 0.65, 0.66, 0.66 and F1 0.49, 0.51, 0.51, 0.51. The curve is flat from about 15 to 22 flights, and the rise is smaller than the spread between folds (about 0.15 in AP). This suggests the limit is the information in the features, not the number of flights. It cannot show whether more flights of the rare fault types (1 elevator flight, 3 rudder flights) would help those types.
- **Seed stability.** The F1 changes by only 0.004 to 0.014 between random seeds, much less than the 0.35 to 0.78 spread between folds, so which flights are held out matters far more than the seed.

## API demo

A small FastAPI service (`src/api.py`) around one random forest trained on all dev flights (`src/18_train_final.py`: 300 trees, threshold 0.5, alarm after 3 raw alarms in a row). You send one flight to `POST /predict`: one row of 21 feature values per second, in time order. It returns each second's fault probability, raw alarm and final alarm, and the first alarm second. `GET /health` lists the model details and the 21 feature names. An optional `threshold` field overrides 0.5. FastAPI also serves a test page at `/docs`.

Tried on one real dev engine-fault flight (`src/19_sample_request.py`): the probability rose from 0.13 to 0.54 in the first fault second, and the alarm followed 2 seconds later, the fastest the 3-in-a-row rule allows. That flight was part of the training data, so this only shows that the service works. It is not a performance result, and the locked test set was not used.

Limits: the service takes the finished feature values, not raw sensor logs, because building those features is done by the table script and is not part of the service. The model is trained on the public ALFA data only. It is a demo: no authentication, one model, loaded from a local file. `tests/test_api.py` checks the service with a tiny model built inside the test, so it needs no data.

## MCP server

`src/mcp_server.py` lets an AI app that speaks the Model Context Protocol (Claude Desktop, Claude Code and others) use the same saved forest as the API. Tools: `model_info`, `list_flights`, `score_flight` (one development flight from `table.csv`) and `score_rows` (new per-second feature rows, the same input as `/predict`).

Before it can score anything it needs the ALFA data and a trained model, which are not in this repository. Download the data (see [Data](#data)), run `src/01_build_table.py`, `src/02_make_split.py` and `src/18_train_final.py`, then `pip install -r requirements.txt -r requirements-mcp.txt`.

The forest was trained on all development flights, so `score_flight` results are in-sample and every reply says `seen_in_training: true`. Locked test flights are refused.

**Claude desktop app.** Use the plugin template in [mcp-plugin/](mcp-plugin/README.md): put your own paths into its `.mcp.json`, zip it as a `.plugin` file and add it under Settings, Plugins, Add, Upload plugin. This is the route that worked with the Windows Store version of the app; that version did not read a `claude_desktop_config.json`.

**Claude Code and other MCP apps.** Register the same command, for example in Claude Code:

```
claude mcp add alfa --env ALFA_DATA=X:\your-folder\data -- X:\your-folder\.venv\Scripts\python.exe X:\your-folder\alfa-fault-detection\src\mcp_server.py
```

Tried through the Claude desktop app on dev engine flight 2018-07-18-15-53-31_1: no alarm before the fault, 15 alarm seconds after it, the first 1.8 s after the fault start. `tests/test_mcp_server.py` checks the tools and is skipped when the mcp package or the saved model is missing.

## Visual guide

Step-by-step GIFs with a caption on every step (about 3 seconds each).

![From raw flight files to the figures](figures/gif2_pipeline.gif)

- GIFs for fig1 to fig8 sit next to each figure above. They show the figure, the real rows the numbers come from, the formula, and why. Drawn by `src/make_gifs_figs.py` from the PNGs and the snippets in `docs/gif_data/` (made by `src/export_fig_data.py`).
- [PowerPoint summary](docs/ALFA_fault_detection_presentation.pptx): 13 slides for a talk (data, pipeline, split, detectors, results, limits, next steps).
- [Time window and label](figures/gif1_time_window.gif): one second, its 5-second window, and when the label flips.
- [How the data is split](figures/gif3_split.gif): the 30 sessions, the five validation rounds, and the locked test part, with the real `split.csv` counts.
- [How the Python files connect](figures/gif4_script_map.gif): which script reads and writes which file.
- [Where the numbers come from](figures/gif5_numbers.gif): real rows of one engine-fault flight, from the raw readings to `mean5`, `std5`, `delta10` and the label.
- [Where r = 0.65 comes from](figures/gif6_fig7_cell.gif): the engine cell of fig7 worked out from real rows.
- [Video of the last two](figures/walkthrough_numbers.mp4)

What is done and what is planned next: [docs/ROADMAP.md](docs/ROADMAP.md).

## Repository layout

- `src/` - the pipeline: `01_build_table.py`, `02_make_split.py`, `03_command_features.py`, `04_final_test.py`,
  `05_figures.py`, plus `evalkit.py` (shared evaluation code) and `config.py` (paths).
- `experiments/` - the exploration and ablation scripts in the order they were run (signal screening with a placebo
  control, baselines, supervised and anomaly models, the frozen-channel investigation). They still contain the original
  absolute paths and are kept as a record of the investigation, not meant to run as they are.
- `src/06_oof_predictions.py`, `src/07_health_figures.py` - out-of-fold predictions on the dev folds, and the health-check figures built from them (run 06 first; it writes `results/`, which is not committed).
- `src/08_correlation.py` - within-flight correlation check: which signals move with the fault label, and which features duplicate each other (figures 6 and 7).
- `src/09_model_comparison.py`, `src/10_comparison_figure.py` - random forest vs CatBoost vs a small Keras network on the dev folds, and the figure (needs the TensorFlow environment in `requirements-tf.txt`).
- `src/12_conformal_threshold.py`, `src/13_conformal_models.py` - alarm threshold chosen from the false-alarm rate on normal seconds (forest only, then all three models), with a per-fold check of the promise.
- `src/14_dummy_baseline.py`, `15_calibration.py`, `16_permutation_importance.py`, `17_learning_curve.py` - the four self-checks on the forest.
- `src/18_train_final.py`, `src/api.py`, `src/19_sample_request.py` - the FastAPI demo: train and save the final forest, the service, and a sample request built from one dev flight (needs `requirements-api.txt`).
- `tests/` - pytest tests for `evalkit.py`, the data split and the API (`test_api.py` needs no data). The tests that need the data are skipped when it is not available. Run: `python -m pytest`.
- `.github/workflows/` - GitHub Actions: runs the tests on every push.
- `figures/` - the plots above, plus the explanation diagrams and GIFs.
- `src/make_diagrams.py`, `src/make_gifs.py`, `src/make_gifs_figs.py`, `src/make_gifs_data.py`, `src/export_gif_data.py` - draw the diagrams and GIFs. The last two use the small data snippets in `docs/gif_data/` (a few public ALFA rows, made by `export_gif_data.py`).
- `docs/` - the full documentation (project report, setup lesson, and a figure by figure explanation of every number with the formulas and code) a code index with line numbers, and the roadmap.

## How to run

1. Download the ALFA processed data from the DOI above and unzip it so that the flight folders are in
   `<data folder>/alfa/processed/processed/carbonZ_*`.
2. `pip install -r requirements.txt`
3. Set the data folder: PowerShell `$env:ALFA_DATA = "X:\your-folder\data"` (default: `data/` inside the repository).
4. Run `src/01_build_table.py`, `02_make_split.py`, `03_command_features.py`, `04_final_test.py`, `05_figures.py` in order.
5. Extra checks (dev folds only): run `src/06` to `src/17` in numeric order. `09` and `10` need the TensorFlow environment (`requirements-tf.txt`); `12` needs the output of `06`, and `13` to `15` need the output of `09`.
6. API demo: `pip install -r requirements-api.txt`, `python src/18_train_final.py`, then `uvicorn api:app --app-dir src` and open `http://127.0.0.1:8000/docs` (add `--port 8011` if port 8000 is busy). `python src/19_sample_request.py` writes a sample request from one dev flight.

## Related work

- **The ALFA dataset.** A. Keipour, M. Mousaei and S. Scherer, "ALFA: A dataset for UAV fault and anomaly detection", The International Journal of Robotics Research 40(2-3), 515-520, 2021. [doi:10.1177/0278364920966642](https://doi.org/10.1177/0278364920966642), [arXiv:1907.06268](https://arxiv.org/abs/1907.06268). Real fixed-wing flights with engine and control-surface faults and ground-truth fault times. All data in this project comes from it.
- **Earlier work by the same authors.** A. Keipour, M. Mousaei and S. Scherer, "Automatic Real-time Anomaly Detection for Autonomous Aerial Vehicles", ICRA 2019, 5679-5685. [doi:10.1109/ICRA.2019.8794286](https://doi.org/10.1109/ICRA.2019.8794286). An online recursive least squares model of related input-output pairs, tested on a 22-flight part of the data.
- **Loading tools.** [castacks/alfa-dataset-tools](https://github.com/castacks/alfa-dataset-tools), the official helper tools for the dataset.
- **DronLomaly.** L. K. Shar et al., "DronLomaly: Runtime detection of anomalous drone behaviors via log analysis and deep learning", APSEC 2022, 119-128. An LSTM trained only on normal flight logs that flags log entries it did not expect, tested on DJI, ArduPilot and PX4 logs.

Numbers reported in these papers use other data subsets and other metrics, so they are not compared with the results here.

## Credits

**Data and paper**

- Keipour, A., Mousaei, M., Scherer, S. "ALFA: A Dataset for UAV Fault and Anomaly Detection." International Journal of Robotics Research 40(2-3):515-520, 2021. https://doi.org/10.1177/0278364920966642
- ALFA dataset, Carnegie Mellon University, licence CC BY 4.0. Not redistributed here, download it from the source: https://doi.org/10.1184/R1/12707963

**Methods**

- Breiman, L. "Random Forests." Machine Learning 45:5-32, 2001.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. "Isolation Forest." IEEE ICDM, 2008.
- Pedregosa et al. "Scikit-learn: Machine Learning in Python." JMLR 12:2825-2830, 2011.

**Background reading on testing ML models**

The checks in this repo are our own code. These sources describe the same ideas:

- scikit-learn, "Common pitfalls and recommended practices" (data leakage, preprocessing): https://scikit-learn.org/stable/common_pitfalls.html
- Made With ML, testing lesson: https://madewithml.com/courses/mlops/testing/
- Eugene Yan, "How to Test Machine Learning Code and Systems": https://eugeneyan.com/writing/testing-ml/
- Jeremy Jordan, "Effective testing for machine learning systems": https://www.jeremyjordan.me/testing-ml/
- Breck et al., "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction," Google, 2017.
- Evidently documentation (data drift; our PSI check is hand-written): https://docs.evidentlyai.com/

**Author and acknowledgements**

- Author: Pham Anh Hoang (GitHub: [PhamAnhHoangg](https://github.com/PhamAnhHoangg))
- AI assistance tool: Claude Sonnet 5.5 (Anthropic)
