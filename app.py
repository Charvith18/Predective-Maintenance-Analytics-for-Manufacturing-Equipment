"""
Machine Failure Prediction & Predictive-Maintenance Dashboard (Streamlit)

Run locally:  streamlit run app.py
Uses the SAME saved pipeline, thresholds and feature definitions as training/evaluation.
All plots/tables shown on the analysis pages are produced by `python -m src.eda` and `python -m src.train`.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_validation import VALID_TYPES
from src.features import FAILURE_MODE_FLAGS, ID_COLUMNS, INPUT_COLUMNS, NUMERIC_RAW
from src.predict import load_artifacts, predict_batch, predict_one

ROOT = Path(__file__).resolve().parent
FIG, TAB = ROOT / "reports" / "figures", ROOT / "reports" / "tables"

st.set_page_config(page_title="Machine Failure Risk", page_icon="🛠️", layout="wide")


# ------------------------------------------------------------------ cached loaders
@st.cache_resource
def get_model():
    return load_artifacts()


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


@st.cache_data
def read_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "machine_failure.csv")


def show_fig(name: str, caption: str = ""):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"Figure {name} not found - run `python -m src.eda` and `python -m src.train`.")


def show_table(name: str, **kw):
    p = TAB / name
    if p.exists():
        st.dataframe(read_csv(name), hide_index=True, **kw)
    else:
        st.info(f"Table {name} not found - run the training/EDA scripts first.")


try:
    model, meta = get_model()
except Exception as e:  # clear message instead of a stack trace
    st.error(f"Could not load the trained model: {e}")
    st.stop()

import sklearn
_trained_with = meta.get("environment", {}).get("scikit_learn")
if _trained_with and _trained_with != sklearn.__version__:
    st.sidebar.warning(f"Model trained with scikit-learn {_trained_with}, running {sklearn.__version__}. "
                       "Retrain (`python -m src.train`) or pin the same version if predictions look wrong.")

PAGES = ["Project overview", "Dataset summary", "Exploratory analysis", "Model comparison",
         "Evaluation", "Explainability", "Predict failure risk", "Limitations & benefits"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{meta['model_name']}** v{meta['model_version']}  \n"
                   f"Decision threshold: **{meta['decision_threshold']:.3f}**  \n"
                   f"Medium-risk threshold: **{meta['medium_risk_threshold']:.3f}**")
st.sidebar.caption("Decision-support prototype - not a safety system.")

# ------------------------------------------------------------------ pages
if page == "Project overview":
    st.title("Machine Failure Prediction")
    st.subheader("Predicting equipment-failure risk from operating measurements")
    st.markdown(f"""
**Goal.** Given one snapshot of a machine's operating conditions, estimate the probability that a failure
is recorded, classify it, and suggest a follow-up action.

**Task.** Binary classification of `Machine failure` (about {meta['dataset']['failure_rate_pct']}% positives).

**Inputs.** Machine type (L/M/H), air temperature, process temperature, rotational speed, torque, tool wear
(+ 3 engineered features: temperature difference, mechanical power, wear x torque).

**Deliberately excluded.** Failure-mode flags (`{', '.join(FAILURE_MODE_FLAGS)}`) because they describe *why* a
failure happened (they leak the answer), and identifiers (`{', '.join(ID_COLUMNS)}`).

**What this is not.** The data has no timestamps or machine histories, so the system does **not** predict
*when* a machine will fail or its remaining useful life.
""")
    m = meta["test_metrics_at_decision_threshold"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test recall", f"{m['recall']:.1%}")
    c2.metric("Test precision", f"{m['precision']:.1%}")
    c3.metric("Test PR-AUC", f"{m['PR_AUC']:.3f}")
    c4.metric("Missed failures (test)", f"{m['FN']} of {m['TP'] + m['FN']}")
    st.caption("Metrics are from a held-out 20% test set used once, after model and threshold selection.")

elif page == "Dataset summary":
    st.header("Dataset summary")
    df = read_data()
    info = json.load(open(TAB / "dataset_summary.json")) if (TAB / "dataset_summary.json").exists() else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}"); c2.metric("Columns", df.shape[1])
    c3.metric("Failures", f"{int(df['Machine failure'].sum())}"); c4.metric("Failure rate", f"{100*df['Machine failure'].mean():.2f}%")
    st.markdown("**Sample records**")
    st.dataframe(df.head(10), hide_index=True)
    st.markdown("**Data quality checks**")
    q = {"Missing values": info.get("missing_values"), "Duplicate rows": info.get("duplicate_rows"),
         "Timestamp column present": info.get("has_timestamp_column"),
         "Repeated machine IDs": (not info.get("no_machine_id_repeats")) if info else None}
    st.table(pd.DataFrame({"check": q.keys(), "result": [str(v) for v in q.values()]}))
    st.markdown("**Numeric summary**"); show_table("numeric_summary.csv")
    st.markdown("**Failure-mode flags (excluded from model inputs)**"); show_table("failure_mode_flags.csv")
    st.caption("Nearly every failure carries a flag, so using flags as inputs would simply hand the model the answer.")

elif page == "Exploratory analysis":
    st.header("Exploratory data analysis")
    st.caption("Cross-sectional analysis only: the data has no time axis, so degradation trends over time cannot be shown. "
               "Patterns below are associations, not proof of cause.")
    a, b = st.columns(2)
    with a: show_fig("01_class_distribution.png")
    with b: show_fig("02_failure_by_type.png")
    show_fig("03_feature_distributions.png"); show_fig("04_boxplots_by_failure.png")
    a, b = st.columns(2)
    with a: show_fig("05_failure_by_tool_wear.png")
    with b: show_fig("06_correlation_heatmap.png")
    show_fig("07_interactions_scatter.png"); show_fig("08_engineered_feature_distributions.png")
    st.subheader("Why these engineered features?")
    st.markdown("Each rule below was checked against the data; where a rule holds, the corresponding failure label is always set.")
    show_table("engineered_feature_rule_check.csv")

elif page == "Model comparison":
    st.header("Model comparison")
    st.markdown("Five-fold stratified CV x 3 repeats on the **training split only**. PR-AUC is the headline metric because "
                "failures are rare; recall/precision here use the default 0.5 cut-off.")
    show_table("model_comparison_cv.csv")
    show_fig("10_cv_stability_boxplot.png")
    st.subheader("Do engineered features help?"); show_table("ablation_engineered_features.csv")
    st.subheader("Hyper-parameter search (includes class weighting options)")
    g = read_csv("grid_search_results.csv"); st.dataframe(g.sort_values(["model", "rank_test_score"]), hide_index=True)
    st.subheader("All models on the test set (reported after selection, at 0.5)"); show_table("all_models_test_metrics_at_0.5.csv")
    st.success(f"Selected model: **{meta['model_name']}** (highest mean CV PR-AUC, {meta['cv_pr_auc_mean']:.3f} +/- {meta['cv_pr_auc_std']:.3f}).")

elif page == "Evaluation":
    st.header("Evaluation on the held-out test set")
    m = meta["test_metrics_at_decision_threshold"]
    cols = st.columns(6)
    for col, (k, lab) in zip(cols, [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("PR_AUC", "PR-AUC"), ("ROC_AUC", "ROC-AUC"), ("accuracy", "Accuracy")]):
        col.metric(lab, f"{m[k]:.3f}")
    st.caption("Accuracy is secondary: a model that never predicts failure would already score ~96.6%.")
    a, b = st.columns(2)
    with a:
        show_fig("12_confusion_matrix_test.png")
        st.markdown(f"**False negatives (missed failures): {m['FN']}**  |  **False positives (unneeded inspections): {m['FP']}**")
    with b:
        show_fig("11_threshold_selection.png", "Threshold chosen on out-of-fold TRAINING predictions, not on test data.")
    show_fig("13_pr_roc_curves_test.png")
    a, b = st.columns(2)
    with a: show_fig("14_calibration.png")
    with b:
        st.markdown("**Risk bands on the test set**"); show_table("risk_bands_test.csv")
        st.caption(f"High = probability >= {meta['decision_threshold']:.3f} (F2-optimal on validation folds). "
                   f"Medium = >= {meta['medium_risk_threshold']:.3f} (training baseline failure rate). Low = below. "
                   "These are documented assumptions, not safety limits.")
    st.subheader("Missed failures (false negatives) on test"); show_table("test_false_negatives.csv")
    st.caption("Most misses are tools with wear above ~200 min but no other warning sign - consistent with failures that "
               "are not determined by the recorded measurements.")
    st.subheader("False alarms (false positives) on test"); show_table("test_false_positives.csv")

elif page == "Explainability":
    st.header("Which inputs drive the predictions?")
    st.warning("Importance describes what the *model* uses, i.e. predictive association - it is not evidence of physical cause.")
    show_fig("15_permutation_importance.png")
    st.markdown("""
* **Left plot (raw inputs):** shuffling an input also breaks every engineered feature built from it. That is why air and
  process temperature look important - they jointly define the temperature difference.
* **Right plot (model features):** shows the engineered features directly: wear x torque, rotational speed,
  temperature difference and power carry most of the signal.
""")
    a, b = st.columns(2)
    with a: show_table("permutation_importance_raw_inputs.csv")
    with b: show_table("permutation_importance_model_features.csv")
    show_fig("16_partial_dependence.png")
    shap_img = FIG / "17_shap_summary.png"
    if shap_img.exists():
        show_fig("17_shap_summary.png", "SHAP summary (generated by src/explain_shap.py)")
    else:
        st.info("SHAP plot not generated. Install `shap` and run `python -m src.explain_shap` (optional).")

elif page == "Predict failure risk":
    st.header("Predict failure risk for one machine")
    ranges = meta["observed_ranges"]
    typical = {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
               "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}
    presets = {"Typical operation": {}, "Worn tool (220 min)": {"Tool wear [min]": 220},
               "High load (torque 65, tool wear 200, type L)": {"Torque [Nm]": 65.0, "Tool wear [min]": 200, "Type": "L"},
               "Small temperature gap, low speed": {"Process temperature [K]": 308.4, "Rotational speed [rpm]": 1300, "Torque [Nm]": 45.0}}
    choice = st.selectbox("Load an example (optional)", list(presets))
    v = {**typical, **presets[choice]}

    with st.form("machine_form"):
        c1, c2, c3 = st.columns(3)
        mtype = c1.selectbox("Machine type", VALID_TYPES, index=VALID_TYPES.index(v["Type"]), key=f"t_{choice}")
        air = c2.number_input("Air temperature [K]", value=float(v["Air temperature [K]"]), step=0.1, format="%.1f", key=f"a_{choice}")
        proc = c3.number_input("Process temperature [K]", value=float(v["Process temperature [K]"]), step=0.1, format="%.1f", key=f"p_{choice}")
        c4, c5, c6 = st.columns(3)
        rpm = c4.number_input("Rotational speed [rpm]", value=float(v["Rotational speed [rpm]"]), step=10.0, key=f"r_{choice}")
        torque = c5.number_input("Torque [Nm]", value=float(v["Torque [Nm]"]), step=0.5, key=f"q_{choice}")
        wear = c6.number_input("Tool wear [min]", value=float(v["Tool wear [min]"]), step=1.0, key=f"w_{choice}")
        go = st.form_submit_button("Predict", type="primary")
    st.caption("Training ranges: " + "; ".join(f"{k} {lo:g}-{hi:g}" for k, (lo, hi) in ranges.items()))

    if go:
        rec = {"Type": mtype, "Air temperature [K]": air, "Process temperature [K]": proc,
               "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear}
        try:
            r = predict_one(rec, model, meta)
        except ValueError as e:
            st.error(f"Invalid input: {e}")
        else:
            for w in r["warnings"]:
                st.warning(w)
            a, b, c = st.columns(3)
            a.metric("Predicted failure probability", f"{r['failure_probability']:.1%}")
            b.metric("Predicted class", r["predicted_class"])
            c.metric("Risk category", r["risk_category"])
            st.progress(min(max(r["failure_probability"], 0.0), 1.0))
            {"High": st.error, "Medium": st.warning, "Low": st.success}[r["risk_category"]](f"**Suggested follow-up:** {r['recommended_action']}")
            if r["watch_conditions"]:
                st.markdown("**Conditions worth checking (patterns seen in this dataset, not safety limits):**")
                for n in r["watch_conditions"]:
                    st.markdown(f"- {n}")
            st.caption("The probability is a model estimate (isotonic-calibrated on training data), not a guarantee. "
                       "Recommendations are decision support, not a maintenance schedule.")

    st.markdown("---")
    st.subheader("Batch prediction from CSV")
    st.caption("Required columns: " + ", ".join(INPUT_COLUMNS) + ". Extra columns (e.g. flags, IDs, the target) are ignored.")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            res = predict_batch(pd.read_csv(up), model, meta)
            for w in res.attrs.get("warnings", []):
                st.warning(w)
            st.dataframe(res)
            st.download_button("Download predictions", res.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        except ValueError as e:
            st.error(f"Invalid file: {e}")

else:
    st.header("Limitations & potential benefits")
    st.subheader("Limitations")
    st.markdown("""
* **Snapshot data only:** no timestamps, machine histories or maintenance logs. No time-to-failure or RUL prediction.
* **Rare class:** only ~3.4% failures, so test-set estimates (68 failures) carry real uncertainty.
* **Possibly synthetic data:** failure labels follow clean rules of temperature gap, power and wear x torque, so real
  plants may be noisier. Performance here may not transfer.
* **Unpredictable failures:** a few failures occur on tools with high wear and no other warning sign; the model misses most of them.
* **Association, not causation.** Importance and recommendations describe patterns in this dataset.
* **Thresholds are assumptions** chosen to favour recall (F2); real costs would change them.
* **Out-of-range inputs** are extrapolations - the app warns about them.
""")
    st.subheader("Potential benefit (hypothetical framework)")
    st.markdown("This dataset has **no cost or downtime data**, so it cannot prove savings. Enter *your own assumed* costs to see how "
                "the confusion matrix on the 2,000 test records *would* translate - it is an illustration, not a result.")
    m = meta["test_metrics_at_decision_threshold"]
    c1, c2, c3 = st.columns(3)
    c_fail = c1.number_input("Assumed cost of an unplanned failure", value=10000.0, step=500.0)
    c_insp = c2.number_input("Assumed cost of one preventive inspection", value=300.0, step=50.0)
    c_fix = c3.number_input("Assumed cost of a failure caught early (planned repair)", value=2000.0, step=100.0)
    no_model = (m["TP"] + m["FN"]) * c_fail
    with_model = m["FN"] * c_fail + m["TP"] * (c_insp + c_fix) + m["FP"] * c_insp
    st.table(pd.DataFrame({"scenario": ["No model (all failures unplanned)", "With model at chosen threshold"],
                           "hypothetical cost": [f"{no_model:,.0f}", f"{with_model:,.0f}"]}))
    st.caption("Cost = FN x failure cost + TP x (inspection + early repair) + FP x inspection. All inputs are user assumptions.")
