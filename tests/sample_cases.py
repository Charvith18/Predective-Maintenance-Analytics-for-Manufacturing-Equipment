"""Generate a table of sample predictions (Step 15):  python -m tests.sample_cases"""
import pandas as pd
from pathlib import Path
from src.predict import load_artifacts, predict_batch

model, meta = load_artifacts()
base = {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}
cases = [("Typical operation", {}),
         ("Fresh tool (wear 0)", {"Tool wear [min]": 0}),
         ("Worn tool (wear 150)", {"Tool wear [min]": 150}),
         ("Very worn tool (wear 220)", {"Tool wear [min]": 220}),
         ("Type L", {"Type": "L"}), ("Type H", {"Type": "H"}),
         ("Overstrain: high torque + wear", {"Type": "L", "Torque [Nm]": 62.0, "Tool wear [min]": 200}),
         ("Small temp gap + low speed", {"Process temperature [K]": 308.4, "Air temperature [K]": 300.0, "Rotational speed [rpm]": 1300, "Torque [Nm]": 45.0}),
         ("Very high power (torque 70, 1500 rpm)", {"Torque [Nm]": 70.0, "Rotational speed [rpm]": 1500}),
         ("Very low power (torque 5, 1600 rpm)", {"Torque [Nm]": 5.0, "Rotational speed [rpm]": 1600}),
         ("Training-range minimum wear/torque", {"Torque [Nm]": 3.8, "Tool wear [min]": 0}),
         ("Training-range maximum wear", {"Tool wear [min]": 253}),
         ("OUT OF RANGE wear (400)", {"Tool wear [min]": 400})]
recs = [{**base, **c[1]} for c in cases]
out = predict_batch(recs, model, meta)
out.insert(0, "case", [c[0] for c in cases])
out["failure_probability"] = out["failure_probability"].round(4)
Path("reports/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("reports/tables/sample_predictions.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(out[["case", "Type", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "failure_probability", "predicted_failure", "risk_category"]].to_string(index=False))
print("warnings:", out.attrs.get("warnings"))
