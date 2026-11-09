# Machine Failure Prediction \& Predictive Maintenance

Predict whether a machine failure is recorded from operating measurements, estimate the failure probability,
explain the drivers, and turn the score into a low / medium / high risk category with a suggested follow-up.

> \*\*Scope.\*\* The data is a single snapshot per row (no timestamps, no machine histories). The system estimates
> it does predict a machine will fail or remaining useful life.

## Folder structure

```
machine\_failure\_prediction/
├── app.py                         # Streamlit dashboard (8 pages + prediction form + CSV batch)
├── requirements.txt  .gitignore  .streamlit/config.toml
├── data/machine\_failure.csv       # the provided dataset (10,000 x 14)
├── models/
│   ├── machine\_failure\_pipeline.joblib   # saved preprocessing + model (calibrated Random Forest)
│   └── model\_metadata.json               # features, thresholds, ranges, metrics, versions
├── src/
│   ├── features.py          # constants + feature engineering (shared by training AND inference)
│   ├── data\_validation.py   # dataset inspection + input validation
│   ├── eda.py               # Steps 3-6: EDA figures/tables
│   ├── train.py             # Steps 7-13: split, tune, CV, threshold, test, importance, save
│   ├── predict.py           # inference, risk categories, recommendations
│   └── explain\_shap.py      # OPTIONAL SHAP (needs `pip install shap`; untested here)
├── tests/                   # test\_pipeline.py (23 unit tests) + sample\_cases.py
├── reports/figures, reports/tables    # every plot/table shown in the app (generated)
└── docs/PROJECT\_REPORT.md  docs/VIVA\_QA.md
```

## Run locally

```bash
python -m venv .venv \&\& source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.eda            # Steps 3-6  -> reports/
python -m src.train          # Steps 7-13 -> models/ + reports/   (a few minutes)
python -m unittest discover -s tests -v                  # Step 15 tests
python -m tests.sample\_cases                             # sample predictions table
streamlit run app.py
```

## Results (held-out test set: 2,000 rows, 68 failures; produced by the code above)

|Metric @ chosen threshold 0.227|Value|
|-|-|
|Recall|0.838 (57 of 68 failures caught, 11 missed)|
|Precision|0.891 (7 false alarms)|
|F1 / F2|0.864 / 0.848|
|PR-AUC / ROC-AUC|0.882 / 0.972|
|Accuracy (secondary)|0.991 (a never-fail model scores 0.966)|

Cross-validated PR-AUC (training data, 5-fold x 3): Random Forest 0.900 ± 0.028, Decision Tree 0.883,
HistGradientBoosting 0.880, Logistic Regression 0.496, baseline 0.034. See `docs/PROJECT\_REPORT.md` for the full analysis.

### Troubleshooting

|Symptom|Likely cause / fix|
|-|-|
|`ModuleNotFoundError: src`|Run from the project root; keep `src/\_\_init\_\_.py`; main file must be `app.py` at repo root.|
|`FileNotFoundError ... machine\_failure\_pipeline.joblib`|Model file not committed (check `.gitignore`/upload) or path changed. Run `python -m src.train` and commit `models/`.|
|Unpickling / `InconsistentVersionWarning` / odd predictions|scikit-learn version differs from training. Keep `scikit-learn==1.8.0`, or retrain in the deployment environment and commit the new `models/`.|
|Dependency-resolver conflict (e.g. pandas/numpy)|Loosen the version bound that conflicts in `requirements.txt` and retrain with those versions.|
|Figures show "not found"|`reports/` was not committed - run `python -m src.eda \&\& python -m src.train` and commit it.|
|App sleeps on Community Cloud|Normal for free apps; wake it with a click.|

## What has been verified

Executed in the build environment: EDA, training, saved-model reload check, 23 unit tests (all pass), sample predictions,
and a smoke test of every Streamlit page using a *stub* of the Streamlit library.


