"""
Inference helpers shared by the Streamlit app, the tests and the command line.

    python -m src.predict --demo

Loads the saved Joblib pipeline (NO retraining) and its metadata (feature names,
thresholds, observed ranges).
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data_validation import validate_input
from .features import add_features  # noqa: F401  (needed so joblib can unpickle the pipeline)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "machine_failure_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

# Tool-strain thresholds seen in THIS dataset (wear x torque, by Type). Data-observed
# association only - NOT an engineering or safety standard.
STRAIN_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


def load_artifacts(model_path=MODEL_PATH, meta_path=META_PATH):
    if not Path(model_path).exists() or not Path(meta_path).exists():
        raise FileNotFoundError(
            f"Model files not found ({model_path}). Run `python -m src.train` from the project root first.")
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def risk_category(p: float, meta: dict) -> str:
    """Low / Medium / High from the thresholds stored in the metadata."""
    if p >= meta["decision_threshold"]:
        return "High"
    if p >= meta["medium_risk_threshold"]:
        return "Medium"
    return "Low"


ACTIONS = {
    "High": "Prioritise this machine for inspection before continued heavy use; check the flagged conditions below "
            "(tool condition, load, cooling/temperatures).",
    "Medium": "Above fleet-average failure risk: schedule an inspection in the next planned maintenance window and "
              "monitor the operating conditions listed below.",
    "Low": "Risk is below the fleet baseline: continue normal operation with routine monitoring.",
}


def watch_conditions(row: pd.Series) -> list[str]:
    """Plain-language notes about conditions that, in this dataset, coincided with failures."""
    notes = []
    diff = row["Process temperature [K]"] - row["Air temperature [K]"]
    power = row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2 * 3.141592653589793 / 60
    strain = row["Tool wear [min]"] * row["Torque [Nm]"]
    if diff < 8.6 and row["Rotational speed [rpm]"] < 1380:
        notes.append(f"Small process-air temperature gap ({diff:.1f} K) at low speed - check cooling/heat dissipation.")
    if power < 3500 or power > 9000:
        notes.append(f"Mechanical power {power:,.0f} W is outside the 3,500-9,000 W band seen for healthy rows - check load/speed.")
    lim = STRAIN_LIMIT[row["Type"]]
    if strain > lim:
        notes.append(f"Tool wear x torque = {strain:,.0f}, above {lim:,} seen for type {row['Type']} failures - review tool condition and load.")
    elif strain > 0.9 * lim:
        notes.append(f"Tool wear x torque = {strain:,.0f} is approaching {lim:,} - consider a tool check.")
    if row["Tool wear [min]"] >= 200:
        notes.append(f"Tool wear {row['Tool wear [min]']:.0f} min: failure rate rises sharply above ~200 min in this data - review tool replacement.")
    return notes


def predict_batch(records, model, meta) -> pd.DataFrame:
    """Validate -> predict. Returns one row per input with probability, class, risk."""
    clean, warns = validate_input(records, meta.get("observed_ranges"))
    proba = model.predict_proba(clean)[:, 1]
    out = clean.copy()
    out["failure_probability"] = proba
    out["predicted_failure"] = (proba >= meta["decision_threshold"]).astype(int)
    out["risk_category"] = [risk_category(p, meta) for p in proba]
    out.attrs["warnings"] = warns
    return out


def predict_one(record: dict, model, meta) -> dict:
    res = predict_batch(record, model, meta)
    row = res.iloc[0]
    return {
        "failure_probability": float(row["failure_probability"]),
        "predicted_failure": int(row["predicted_failure"]),
        "predicted_class": "Failure" if row["predicted_failure"] == 1 else "No failure",
        "risk_category": row["risk_category"],
        "recommended_action": ACTIONS[row["risk_category"]],
        "watch_conditions": watch_conditions(row),
        "warnings": res.attrs["warnings"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    model, meta = load_artifacts()
    demo = {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}
    print(json.dumps(predict_one(demo, model, meta), indent=2))
