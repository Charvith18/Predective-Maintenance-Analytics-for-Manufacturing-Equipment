# Project Report: AI-Based Machine Failure Prediction and Predictive Maintenance

All numbers below were produced by the code in this repository (`src/eda.py`, `src/train.py`, `tests/`). The random seed is 42.

## 0. Dataset inspection and suitability
**`machine_failure.csv`** - 10,000 rows x 14 columns; no missing values; no duplicate rows (also none on the feature columns).

| Column | Role |
|---|---|
| `UDI`, `Product ID` | Identifiers (unique per row) - excluded |
| `Type` (L 6,000 / M 2,997 / H 1,003) | Input, categorical |
| Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] | Inputs, numeric |
| `Machine failure` | **Target** (binary): 339 failures = 3.39%, 9,661 non-failures |
| `TWF, HDF, PWF, OSF, RNF` | Failure-mode flags - **excluded (leakage)** |

**`submission.csv`** - 90,954 rows; columns `id` and `Machine failure`; values are continuous probabilities, `id` runs 136,429-227,382 and
does not overlap the main file's `UDI` (1-10,000); there are no features and no true labels. It looks like a competition-style
sample/prediction file for a separate test set. It cannot be used for training or evaluation and was not used.

**Suitable for:** snapshot-based binary failure-risk classification, model comparison, explainability.
**Not suitable for:** time-to-failure/RUL, degradation trends, cost or downtime claims (no timestamps, histories, costs).
**Provenance:** the structure resembles the public "AI4I 2020 Predictive Maintenance" dataset, which is synthetic as far as I recall; this was
not verified, so state the real source yourself. The data behaves as if labels come from rules (see Step 6), which supports the synthetic reading.

## Step 1 - Objective
Binary classification of `Machine failure` from `Type` + five sensor/usage inputs (+ three engineered features). Outputs: probability,
class, risk category (Low/Medium/High), follow-up suggestion. Motivation: a missed failure usually costs more than an extra inspection,
so evaluation emphasises recall and PR-AUC. Out of scope: failure timing, RUL, cost savings, causal claims.

## Step 2 - Data collection
Only the provided CSV is used. Missing for a real deployment: timestamps, machine IDs that repeat, maintenance and repair logs, downtime
and cost records, sensor time series, operating shifts/environment, failure-mode labelling process.

## Step 3 - Historical failure patterns (associations only)
* Failure rate by type: L 3.92% (235/6,000), M 2.77% (83/2,997), H 2.09% (21/1,003).
* Mean torque: 39.6 Nm (no failure) vs 50.2 Nm (failure); mean tool wear 106.7 vs 143.8 min; mean rotational speed 1540 vs 1496 rpm; air temperature 299.97 vs 300.89 K.
* Failure rate by tool wear: ~2.2-2.9% up to 200 min, 13.8% (200-220), 22.7% (220-240), 36.4% (240-253, only 11 rows).
* Spearman correlation with the target is weak for every single input (largest: torque 0.17, speed -0.17), i.e. failures depend on
  *combinations* of inputs, which is why a linear model does badly (Step 8).
Figures: `reports/figures/01-08*.png`. Correlation does not show causation.

## Step 4 - Cleaning and consistency checks
* No missing values/duplicates; Type has exactly L/M/H; the first letter of Product ID always equals Type.
* Temperatures are plausible (process is 7.6-12.1 K above air), speeds 1,168-2,886 rpm, torque 3.8-76.6 Nm, wear 0-253 min.
* IQR outliers: rotational speed 418 rows (skew 2.0), torque 69 rows; none elsewhere. **They were kept**: they are valid readings, and the 69
  torque outliers have an 89.9% failure rate - removing them would delete real failure signal.
* Preprocessing in the pipeline: median imputation (a safeguard for future inputs), one-hot encoding of `Type` (unknown-safe),
  standard scaling only for logistic regression.

## Step 5 - Trend analysis
No timestamps/machine histories exist (every Product ID is unique; tool wear decreases 119 times when rows are read in UDI order, so row order is not a time axis).
Therefore only cross-sectional analysis is possible; **degradation trends and time-to-failure cannot be established from this data.**

## Step 6 - Feature engineering
| Feature | Definition | Why |
|---|---|---|
| `temp_diff_K` | process temp - air temp | heat-dissipation proxy |
| `power_W` | torque x rpm x 2pi/60 | mechanical power |
| `wear_torque` | tool wear x torque | overstrain proxy |

Check against the data (`reports/tables/engineered_feature_rule_check.csv`): every one of the 115 `HDF` rows has temp_diff < 8.6 K and speed < 1,380 rpm
and vice versa; all 95 `PWF` rows have power < 3,500 W or > 9,000 W and vice versa; all 98 `OSF` rows have wear x torque above 11,000/12,000/13,000 (L/M/H) and vice versa.
So the features are physically meaningful *and* empirically justified. Ablation (CV PR-AUC, raw -> with engineered): Random Forest 0.738 -> 0.900,
Decision Tree 0.670 -> 0.883, HistGradientBoosting 0.807 -> 0.880, Logistic Regression 0.461 -> 0.496. All three are computed from the raw inputs at inference
time inside the same pipeline.

## Step 7 - Split and leakage prevention
Stratified 80/20 split (seed 42): train 8,000 rows (271 failures, 3.388%), test 2,000 rows (68 failures, 3.400%). The test set is used once at the end.
Failure-mode flags are **not** inputs: 330 of 339 failures carry a flag, so they would give away the label. Flags are used only for error analysis after prediction.
Imputer/scaler/encoder live inside the Pipeline, so they are fitted on training folds only. No grouping variable exists, so group-based CV is not applicable.
No resampling (SMOTE) was used; class weighting was tried as a hyper-parameter instead.

## Step 8 - Models
Baseline (always predict class prior), Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting. Each tuned by grid search
(5-fold stratified CV on training data, PR-AUC; grids include `class_weight`). Result: class weighting did not improve PR-AUC for the best configurations
(unweighted won for every model), see `grid_search_results.csv`. Chosen settings are in `tuned_hyperparameters.json`.

## Step 9 - Stability (5-fold x 3 repeats, training data)
| Model | PR-AUC mean ± sd | Recall @0.5 | Precision @0.5 | Train PR-AUC (overfit gap) |
|---|---|---|---|---|
| **Random Forest** | **0.900 ± 0.028** | 0.805 | 0.983 | 0.975 (0.075) |
| Decision Tree | 0.883 ± 0.030 | 0.797 | 0.959 | 0.929 (0.046) |
| HistGradientBoosting | 0.880 ± 0.034 | 0.768 | 0.940 | 1.000 (0.120) |
| Logistic Regression | 0.496 ± 0.054 | 0.239 | 0.754 | 0.508 (0.013: under-fits) |
| Baseline | 0.034 ± 0.000 | 0 | 0 | - |

The top three are within about one standard deviation of each other, so the ranking among them is not decisive. The Random Forest was selected for the highest mean
PR-AUC and the smallest gap among the tree ensembles. Logistic Regression's low score shows failure depends on non-linear thresholds/interactions.

## Step 10 - Evaluation, errors, threshold
* **Calibration:** isotonic calibration lowered out-of-fold Brier score from 0.00764 to 0.00659, so the saved model is the calibrated Random Forest and its
  output is a reasonable probability estimate (not a guarantee).
* **Threshold (validation data only):** maximise F2 (recall weighted more than precision, because a missed failure is assumed costlier) on out-of-fold
  training predictions, taking the middle of the near-optimal plateau -> **0.2267**. Not tuned on the test set.
* **Test results @0.2267:** TP 57, FP 7, FN 11, TN 1,925; precision 0.891, recall 0.838, F1 0.864, F2 0.848, PR-AUC 0.882, ROC-AUC 0.972, accuracy 0.991.
  At the default 0.5: precision 0.947, recall 0.794 (54 TP, 3 FP, 14 FN). The never-fail baseline has accuracy 0.966 and recall 0.
* **For transparency** (reported after selection, not used to choose): test PR-AUC Random Forest 0.894 (uncalibrated), HistGradientBoosting 0.903, Decision Tree 0.874,
  Logistic Regression 0.464. HistGradientBoosting scoring slightly higher on test than the selected model is within the noise expected from 68 test failures.
* **Missed failures (11):** by flag (analysis only) 8 are tool-wear failures and 1 a heat failure; 2 carry no flag. Missed ones are mostly tools with 198-235 min wear and no other
  warning sign, i.e. failures the recorded measurements barely explain. This is a hard ceiling for any model on this data.
* **False alarms (7):** mostly small temperature gap with low speed or very high torque but no failure recorded.

## Step 11 - Interpretation (association, not causation)
Permutation importance (drop in test PR-AUC, 20 repeats). On model features: wear_torque 0.251, rotational speed 0.230, temp_diff 0.216, power 0.176,
tool wear 0.030, others near 0. On raw inputs, torque 0.76, speed 0.67, air temperature 0.57, tool wear 0.37, process temperature 0.32, type 0.05 - air and process
temperature appear because shuffling either destroys `temp_diff_K`. These relationships are physically plausible (overload, heat dissipation, tool wear) but the
model shows association only. A SHAP script is provided (`src/explain_shap.py`) but could not be run where this was built; permutation importance and partial dependence were run instead.

## Step 12 - Risk bands and maintenance insights
Low < 0.0339 (training baseline failure rate) <= Medium < 0.2267 <= High. These are documented assumptions, not safety limits.
Test set: Low 1,769 rows with 4 failures (0.23%); Medium 167 rows, 7 failures (4.19%); High 64 rows, 57 failures (89.1%).
Suggested actions: High - prioritise inspection and check flagged conditions; Medium - inspect in the next planned window; Low - routine monitoring. The app also lists
data-observed conditions (small temperature gap at low speed, power outside 3.5-9 kW, wear x torque above 11/12/13k, wear >= 200 min). These are decision-support hints, not a guaranteed schedule.

## Step 13 - Saved artifacts
`models/machine_failure_pipeline.joblib` (calibrated pipeline) and `models/model_metadata.json` (version, seed, inputs, excluded columns, thresholds, training ranges,
hyper-parameters, test metrics, library versions). `train.py` reloads the file and asserts identical predictions; a unit test recomputes the stored test PR-AUC from the saved model.

## Step 14 - Streamlit application
`app.py`: overview, dataset summary, EDA, model comparison, evaluation (metrics, confusion matrix, curves, risk bands, errors), explainability, prediction form (probability,
class, risk, action, warnings), CSV batch prediction, limitations and a hypothetical benefit calculator. The form uses exactly `INPUT_COLUMNS`; flags/IDs/target are never inputs (extra CSV columns are ignored).
Predictions come from the saved pipeline; nothing is hard-coded.

## Step 15 - Tests
`python -m unittest discover -s tests -v`: 23 tests, all passing. They cover leakage columns, engineered features, thresholds, reproduction of the stored test PR-AUC, typical/boundary/wear/type cases,
case-insensitive `Type`, batch = single predictions, determinism, out-of-range warnings, and invalid inputs (bad type, negative/zero/NaN/inf/text values, missing columns).
`reports/tables/sample_predictions.csv` lists 13 sample cases: for example typical operation 0.02% (Low); wear 220 min 4.0% (Medium); L-type torque 62 Nm + wear 200 min 100% (High);
small temperature gap with 1,300 rpm 61% (High); wear 400 min (out of range) triggers a warning. Predictions are probabilistic; the tests do not require any specific input to be flagged.
The Streamlit pages were smoke-tested only with a stub of the Streamlit library (real Streamlit could not be installed offline).

## Step 16 - Deployment
See `README.md`. Deployment on Streamlit Community Cloud has **not** been performed or verified.

## Step 17 - Potential downtime benefits (hypothetical)
Earlier identification of high-risk machines could let a team prioritise inspections, replace worn tools before heavy use and plan work in maintenance windows. In the test set,
the High band holds 57 of 68 failures in 64 rows (3.2% of records), which would concentrate inspection effort. Assumptions: the deployed machines resemble this data, readings are
available before failure, and inspections actually prevent failures. Risks: false alarms, missed failures (11 of 68 here), distribution shift, over-reliance on the score.
**This dataset contains no cost or downtime data, so no saving is claimed.** Framework: Cost(no model) = N_fail x C_unplanned; Cost(model) = FN x C_unplanned + TP x (C_inspect + C_early_repair) + FP x C_inspect.
The app's calculator lets you enter your own assumed costs; its output is illustrative only.

## Limitations and future work
Snapshot data only; possibly synthetic with rule-like labels; few failures (68 in test) so metrics are uncertain (a confidence interval or repeated splits would help); unpredictable tool-wear failures;
thresholds depend on assumed costs. Future: real time-series sensors and maintenance logs, survival/RUL models, cost-sensitive thresholds, drift monitoring, SHAP review, external validation data.
