"""
Steps 7-13: split, train, validate, evaluate, interpret, save.
Run from the project root:   python -m src.train

Design rules (leakage prevention):
  * Inputs are ONLY Type + 5 sensor columns (+ 3 engineered). No flags, no IDs.
  * One stratified 80/20 split. The 20% TEST set is used exactly once, at the end.
  * Imputation / scaling / one-hot encoding live INSIDE the Pipeline, so they are
    fitted on training folds only.
  * Class imbalance handled with class_weight (no resampling of test/validation data).
  * Hyper-parameters, model choice and decision threshold use TRAINING data only
    (cross-validation / out-of-fold predictions).
"""
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_validation import inspect_dataset, observed_ranges
from .features import (CATEGORICAL, ENGINEERED, FAILURE_MODE_FLAGS, ID_COLUMNS, INPUT_COLUMNS, NUMERIC_RAW,
                       TARGET, add_features, numeric_columns, select_inputs)

warnings.filterwarnings("ignore")
SEED = 42
MODEL_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "machine_failure.csv"
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------- pipeline builder
def build_pipeline(estimator, use_engineered=True, scale=False):
    """features -> (impute, scale, one-hot) -> model, all inside one Pipeline."""
    feat = FunctionTransformer(add_features if use_engineered else select_inputs, validate=False)
    num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    prep = ColumnTransformer([
        ("num", Pipeline([(f"s{i}", s) for i, s in enumerate(num)]), numeric_columns(use_engineered)),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("features", feat), ("prep", prep), ("model", estimator)])


# name -> (estimator, grid, needs_scaling)   (grid keys are prefixed with 'model__')
def candidate_models():
    return {
        "Baseline (always predict prior)": (DummyClassifier(strategy="prior"), {}, False),
        "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=SEED),
                                {"model__C": [0.1, 1, 10], "model__class_weight": [None, "balanced"]}, True),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                          {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 20],
                           "model__class_weight": [None, "balanced"]}, False),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                          {"model__max_depth": [8, 12], "model__min_samples_leaf": [1, 5],
                           "model__class_weight": [None, "balanced_subsample"]}, False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
                                 {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
                                  "model__class_weight": [None, "balanced"]}, False),
    }


# ----------------------------------------------------------------------------- metrics helpers
def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "f2": fbeta_score(y, pred, beta=2),
            "accuracy": accuracy_score(y, pred), "PR_AUC": average_precision_score(y, proba),
            "ROC_AUC": roc_auc_score(y, proba), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def choose_threshold(y, proba, beta=2.0, plateau=0.005):
    """Pick the decision threshold on OUT-OF-FOLD training predictions.

    Objective: maximise F-beta with beta=2 (recall counts 4x as much as precision in the
    harmonic mean), because a missed failure is assumed costlier than an extra inspection.
    To avoid an over-fitted knife-edge value we take the MIDDLE of all thresholds whose F2
    is within `plateau` of the best F2.
    """
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
    good = thr[f >= f.max() - plateau]
    return float(np.median(good)), float(f.max())


# ----------------------------------------------------------------------------- main
def main():
    for d in (MODELS, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)

    # ---- load + sanity checks ----
    df = pd.read_csv(DATA)
    info = inspect_dataset(df)
    assert info["target_is_binary"] and info["missing_values"] == 0
    X, y = df[INPUT_COLUMNS], df[TARGET]
    assert not set(X.columns) & set(FAILURE_MODE_FLAGS + ID_COLUMNS + [TARGET]), "leakage column in inputs!"

    # ---- Step 7: stratified, reproducible split ----
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    split_tab = pd.DataFrame({"records": [len(y_tr), len(y_te)], "failures": [int(y_tr.sum()), int(y_te.sum())]},
                             index=["train", "test"])
    split_tab["failure_rate_pct"] = (100 * split_tab.failures / split_tab.records).round(3)
    split_tab.to_csv(TAB / "train_test_split.csv")
    print(split_tab)

    # ---- Step 8: tune each candidate with 5-fold stratified CV on TRAIN only (PR-AUC) ----
    tune_cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tuned, best_params, grid_logs = {}, {}, []
    for name, (est, grid, scale) in candidate_models().items():
        pipe = build_pipeline(est, True, scale)
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=tune_cv, n_jobs=1, refit=False)
            gs.fit(X_tr, y_tr)
            params = gs.best_params_
            g = pd.DataFrame(gs.cv_results_)
            g.insert(0, "model", name)
            grid_logs.append(g[["model", "params", "mean_test_score", "std_test_score", "rank_test_score"]])
        else:
            params = {}
        best_params[name] = {k.replace("model__", ""): v for k, v in params.items()}
        tuned[name] = (clone(pipe).set_params(**params), scale, est)
        print(f"tuned {name}: {best_params[name]}")
    json.dump(best_params, open(TAB / "tuned_hyperparameters.json", "w"), indent=2, default=str)
    pd.concat(grid_logs).rename(columns={"mean_test_score": "cv_PR_AUC_mean", "std_test_score": "cv_PR_AUC_std"}) \
        .round(4).to_csv(TAB / "grid_search_results.csv", index=False)

    # ---- Step 9: stability: 5-fold x 3 repeats, different seed from tuning ----
    stab_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    scoring = {"PR_AUC": "average_precision", "ROC_AUC": "roc_auc", "recall": "recall",
               "precision": "precision", "f1": "f1", "accuracy": "accuracy"}
    fold_rows, summary = [], []
    for name, (pipe, scale, est) in tuned.items():
        res = cross_validate(pipe, X_tr, y_tr, cv=stab_cv, scoring=scoring, return_train_score=True, n_jobs=1)
        for i in range(len(res["test_PR_AUC"])):
            fold_rows.append({"model": name, "fold": i, **{m: res[f"test_{m}"][i] for m in scoring},
                              "train_PR_AUC": res["train_PR_AUC"][i]})
        row = {"model": name}
        for m in scoring:
            row[f"{m}_mean"], row[f"{m}_std"] = res[f"test_{m}"].mean(), res[f"test_{m}"].std()
        row["train_PR_AUC_mean"] = res["train_PR_AUC"].mean()
        row["overfit_gap_PR_AUC"] = row["train_PR_AUC_mean"] - row["PR_AUC_mean"]
        summary.append(row)
    cv_folds = pd.DataFrame(fold_rows)
    cv_summary = pd.DataFrame(summary).sort_values("PR_AUC_mean", ascending=False)
    cv_folds.to_csv(TAB / "cv_fold_scores.csv", index=False)
    cv_summary.round(4).to_csv(TAB / "model_comparison_cv.csv", index=False)
    print(cv_summary[["model", "PR_AUC_mean", "PR_AUC_std", "recall_mean", "precision_mean", "overfit_gap_PR_AUC"]].round(4))

    plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=cv_folds, x="model", y="PR_AUC", color="#9ecae1")
    sns.stripplot(data=cv_folds, x="model", y="PR_AUC", color="k", size=3)
    plt.xticks(rotation=20); plt.title("Step 9: PR-AUC across 15 CV folds (5-fold x 3 repeats, training data)")
    plt.tight_layout(); plt.savefig(FIG / "10_cv_stability_boxplot.png", dpi=130); plt.close()

    # ---- ablation: do engineered features help? (same tuned params, raw-only inputs) ----
    abl = []
    for name, (pipe, scale, est) in tuned.items():
        if name.startswith("Baseline"):
            continue
        params = {k: v for k, v in pipe.get_params().items() if k.startswith("model__")}
        raw_pipe = build_pipeline(clone(est), use_engineered=False, scale=scale).set_params(**{k: v for k, v in params.items() if k.count("__") == 1})
        r = cross_validate(raw_pipe, X_tr, y_tr, cv=stab_cv, scoring={"PR_AUC": "average_precision", "recall": "recall"}, n_jobs=1)
        e = cv_summary.set_index("model").loc[name]
        abl.append({"model": name, "PR_AUC raw features": r["test_PR_AUC"].mean(), "PR_AUC + engineered": e["PR_AUC_mean"],
                    "recall(0.5) raw": r["test_recall"].mean(), "recall(0.5) + engineered": e["recall_mean"]})
    abl = pd.DataFrame(abl)
    abl.round(4).to_csv(TAB / "ablation_engineered_features.csv", index=False)
    print(abl.round(4))

    # ---- model selection: highest mean CV PR-AUC (training data only) ----
    best_name = cv_summary.iloc[0]["model"]
    best_pipe = tuned[best_name][0]
    print("SELECTED:", best_name, best_params[best_name])

    # ---- calibration decision (still training data only, out-of-fold) ----
    oof_cv = StratifiedKFold(5, shuffle=True, random_state=123)
    oof_raw = cross_val_predict(clone(best_pipe), X_tr, y_tr, cv=oof_cv, method="predict_proba")[:, 1]
    cal_candidate = CalibratedClassifierCV(clone(best_pipe), method="isotonic", cv=5)
    oof_cal = cross_val_predict(cal_candidate, X_tr, y_tr, cv=oof_cv, method="predict_proba")[:, 1]
    brier_raw, brier_cal = brier_score_loss(y_tr, oof_raw), brier_score_loss(y_tr, oof_cal)
    use_cal = brier_cal < brier_raw
    oof = oof_cal if use_cal else oof_raw
    print(f"Brier raw={brier_raw:.5f} calibrated={brier_cal:.5f} -> calibrated used: {use_cal}")

    # ---- Step 10: threshold chosen on OOF training predictions (never on test) ----
    thr_dec, best_f2 = choose_threshold(y_tr, oof, beta=2.0)
    # "Medium risk" lower edge = the baseline failure rate of the TRAINING data (documented assumption):
    # a record scored at or above the fleet-average failure rate is "above-average risk".
    thr_low = float(y_tr.mean())
    thr_low = min(thr_low, thr_dec / 2)
    print(f"decision threshold={thr_dec:.4f} (OOF F2={best_f2:.4f}); medium-risk lower edge={thr_low:.4f}")

    sweep = pd.DataFrame([metrics_at(y_tr, oof, t) for t in np.linspace(0.01, 0.99, 99)])
    sweep.round(4).to_csv(TAB / "threshold_sweep_oof_train.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in ["precision", "recall", "f1", "f2"]:
        ax.plot(sweep.threshold, sweep[m], label=m)
    ax.axvline(thr_dec, color="k", ls="--", label=f"chosen = {thr_dec:.2f}")
    ax.axvline(thr_low, color="grey", ls=":", label=f"medium-risk edge = {thr_low:.2f}")
    ax.set_xlabel("Probability threshold"); ax.set_title("Threshold selection on out-of-fold TRAIN predictions (not test)")
    ax.legend(); plt.tight_layout(); plt.savefig(FIG / "11_threshold_selection.png", dpi=130); plt.close()

    # ---- Final fit on TRAIN, single evaluation on TEST ----
    final = CalibratedClassifierCV(clone(best_pipe), method="isotonic", cv=5) if use_cal else clone(best_pipe)
    final.fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    test_main = metrics_at(y_te, p_te, thr_dec)
    test_default = metrics_at(y_te, p_te, 0.5)
    print("TEST @ chosen threshold:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test_main.items()})

    # transparency table: every tuned model on test (reported AFTER selection; not used to choose anything)
    rows, curves = [], {}
    for name, (pipe, scale, est) in tuned.items():
        m = clone(pipe).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        r = metrics_at(y_te, p, 0.5)
        r["model"] = name
        rows.append(r)
        curves[name] = p
    all_test = pd.DataFrame(rows)[["model", "threshold", "precision", "recall", "f1", "accuracy", "PR_AUC", "ROC_AUC", "TP", "FP", "FN", "TN"]]
    all_test.round(4).to_csv(TAB / "all_models_test_metrics_at_0.5.csv", index=False)
    final_rows = pd.DataFrame([{"setting": f"FINAL {best_name} @ chosen threshold", **test_main},
                               {"setting": f"FINAL {best_name} @ 0.5", **test_default}])
    final_rows.round(4).to_csv(TAB / "final_model_test_metrics.csv", index=False)

    band = np.where(p_te >= thr_dec, "High", np.where(p_te >= thr_low, "Medium", "Low"))
    bt = pd.DataFrame({"band": band, "failure": y_te.values}).groupby("band").failure.agg(records="count", failures="sum")
    bt["failure_rate_pct"] = (100 * bt.failures / bt.records).round(2)
    bt = bt.loc[[b_ for b_ in ["Low", "Medium", "High"] if b_ in bt.index]]
    bt.to_csv(TAB / "risk_bands_test.csv")
    print(bt)

    cm = confusion_matrix(y_te, (p_te >= thr_dec).astype(int))
    pd.DataFrame(cm, index=["actual no failure", "actual failure"], columns=["pred no failure", "pred failure"]).to_csv(TAB / "confusion_matrix_test.csv")
    plt.figure(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["No failure", "Failure"], yticklabels=["No failure", "Failure"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Test confusion matrix (threshold {thr_dec:.2f})")
    plt.tight_layout(); plt.savefig(FIG / "12_confusion_matrix_test.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, p in curves.items():
        if name.startswith("Baseline"):
            continue
        pr, rc, _ = precision_recall_curve(y_te, p); axes[0].plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, p):.3f})")
        fpr, tpr, _ = roc_curve(y_te, p); axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    axes[0].axhline(y_te.mean(), color="grey", ls=":", label="no-skill"); axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].set_title("Precision-Recall (test)")
    axes[1].plot([0, 1], [0, 1], color="grey", ls=":"); axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate"); axes[1].set_title("ROC (test)")
    for a in axes: a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "13_pr_roc_curves_test.png", dpi=130); plt.close()

    fr, mp = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 4.5)); plt.plot(mp, fr, "o-", label="final model"); plt.plot([0, 1], [0, 1], "k:", label="perfect")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed failure fraction"); plt.title("Calibration (test, quantile bins)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "14_calibration.png", dpi=130); plt.close()

    # error analysis: which failures were missed / which alarms were false?
    err = X_te.copy(); err["actual"] = y_te.values; err["proba"] = p_te
    err = pd.concat([err, df.loc[X_te.index, FAILURE_MODE_FLAGS]], axis=1)  # flags used for ANALYSIS only
    err["pred"] = (p_te >= thr_dec).astype(int)
    err[(err.actual == 1) & (err.pred == 0)].sort_values("proba").to_csv(TAB / "test_false_negatives.csv")
    err[(err.actual == 0) & (err.pred == 1)].sort_values("proba", ascending=False).to_csv(TAB / "test_false_positives.csv")
    fn_flags = err[(err.actual == 1) & (err.pred == 0)][FAILURE_MODE_FLAGS].sum().to_dict()
    print("Missed failures by mode flag (analysis only):", fn_flags)

    # ---- Step 11: interpretability (on TEST, permutation, PR-AUC drop) ----
    pi = permutation_importance(final, X_te, y_te, scoring="average_precision", n_repeats=20, random_state=SEED, n_jobs=1)
    imp_raw = pd.DataFrame({"feature": INPUT_COLUMNS, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    imp_raw.round(4).to_csv(TAB / "permutation_importance_raw_inputs.csv", index=False)

    # same idea on the engineered feature space, using an uncalibrated twin fitted on the same TRAIN data
    twin = clone(best_pipe).fit(X_tr, y_tr)
    Xt_te = twin[:-1].transform(X_te)
    names = list(twin.named_steps["prep"].get_feature_names_out())
    names = [n.replace("num__", "").replace("cat__", "") for n in names]
    pi2 = permutation_importance(twin[-1], Xt_te, y_te, scoring="average_precision", n_repeats=20, random_state=SEED, n_jobs=1)
    imp_eng = pd.DataFrame({"feature": names, "importance_mean": pi2.importances_mean, "importance_std": pi2.importances_std}).sort_values("importance_mean", ascending=False)
    imp_eng.round(4).to_csv(TAB / "permutation_importance_model_features.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, d, t in zip(axes, [imp_raw, imp_eng], ["Raw inputs (full model)", "Model features incl. engineered"]):
        d = d.iloc[::-1]; ax.barh(d.feature, d.importance_mean, xerr=d.importance_std, color="#4C72B0"); ax.set_title(f"Permutation importance - {t}"); ax.set_xlabel("Drop in PR-AUC when shuffled")
    plt.tight_layout(); plt.savefig(FIG / "15_permutation_importance.png", dpi=130); plt.close()

    # Partial dependence (manual, works for any pipeline): average predicted risk as one input varies
    top = imp_raw.feature.head(4).tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(4.2 * len(top), 3.8))
    for ax, c in zip(np.atleast_1d(axes), top):
        grid = np.linspace(X_tr[c].quantile(0.01), X_tr[c].quantile(0.99), 25)
        vals = []
        for g in grid:
            Xg = X_te.copy(); Xg[c] = g; vals.append(final.predict_proba(Xg)[:, 1].mean())
        ax.plot(grid, vals); ax.set_xlabel(c); ax.set_ylabel("mean predicted P(failure)")
    plt.suptitle("Partial dependence (association with the model's output, not causation)", y=1.02)
    plt.tight_layout(); plt.savefig(FIG / "16_partial_dependence.png", dpi=130, bbox_inches="tight"); plt.close()

    # ---- Step 13: save pipeline + metadata ----
    joblib.dump(final, MODELS / "machine_failure_pipeline.joblib")
    meta = {
        "model_version": MODEL_VERSION, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": best_name, "calibrated_isotonic": bool(use_cal), "best_hyperparameters": best_params[best_name],
        "random_seed": SEED, "split": {"test_size": 0.2, "stratified": True},
        "input_columns": INPUT_COLUMNS, "numeric_inputs": NUMERIC_RAW, "categorical_inputs": CATEGORICAL,
        "engineered_features": {"temp_diff_K": "Process temp - Air temp", "power_W": "Torque * rpm * 2*pi/60",
                                "wear_torque": "Tool wear * Torque"},
        "excluded_columns": {"leakage_flags": FAILURE_MODE_FLAGS, "identifiers": ID_COLUMNS, "target": TARGET},
        "type_levels": ["L", "M", "H"], "observed_ranges": observed_ranges(df.loc[X_tr.index]),
        "decision_threshold": thr_dec, "medium_risk_threshold": thr_low, "medium_risk_rule": "probability >= training baseline failure rate (documented assumption)",
        "threshold_objective": "max F2 on out-of-fold training predictions (middle of plateau within 0.005 of best)",
        "test_metrics_at_decision_threshold": test_main,
        "cv_pr_auc_mean": float(cv_summary.iloc[0]["PR_AUC_mean"]), "cv_pr_auc_std": float(cv_summary.iloc[0]["PR_AUC_std"]),
        "brier_oof_uncalibrated": brier_raw, "brier_oof_calibrated": brier_cal,
        "dataset": {"rows": info["rows"], "failure_rate_pct": info["failure_rate_pct"]},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    json.dump(meta, open(MODELS / "model_metadata.json", "w"), indent=2, default=float)

    # round-trip check: reload and compare
    reloaded = joblib.load(MODELS / "machine_failure_pipeline.joblib")
    assert np.allclose(reloaded.predict_proba(X_te)[:, 1], p_te), "reloaded model differs!"
    print("Saved + reload check OK.")


if __name__ == "__main__":
    main()
