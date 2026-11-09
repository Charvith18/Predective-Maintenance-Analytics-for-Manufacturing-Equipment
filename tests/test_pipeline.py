"""
Step 15 tests.  Run from the project root:   python -m unittest discover -s tests -v
They check the SAVED pipeline (no retraining) and the input validation.
They do not assert that a given input must be classed as failure - predictions are probabilistic.
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.features import FAILURE_MODE_FLAGS, ID_COLUMNS, INPUT_COLUMNS, TARGET, add_features
from src.predict import load_artifacts, predict_batch, predict_one, risk_category
from src.data_validation import validate_input

MODEL, META = load_artifacts()
TYPICAL = {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
           "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}


def case(**kw):
    d = dict(TYPICAL); d.update(kw); return d


class TestArtifacts(unittest.TestCase):
    def test_inputs_have_no_leakage_columns(self):
        self.assertFalse(set(META["input_columns"]) & set(FAILURE_MODE_FLAGS + ID_COLUMNS + [TARGET]))
        self.assertEqual(META["input_columns"], INPUT_COLUMNS)

    def test_thresholds_ordered(self):
        self.assertTrue(0 < META["medium_risk_threshold"] < META["decision_threshold"] < 1)

    def test_engineered_features_correct(self):
        f = add_features(pd.DataFrame([TYPICAL]))
        self.assertAlmostEqual(f.loc[0, "temp_diff_K"], 10.0)
        self.assertAlmostEqual(f.loc[0, "power_W"], 40 * 1500 * 2 * np.pi / 60)
        self.assertAlmostEqual(f.loc[0, "wear_torque"], 4000.0)

    def test_saved_model_reproduces_reported_test_pr_auc(self):
        df = pd.read_csv("data/machine_failure.csv")
        _, X_te, _, y_te = train_test_split(df[INPUT_COLUMNS], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=META["random_seed"])
        ap = average_precision_score(y_te, MODEL.predict_proba(X_te)[:, 1])
        self.assertAlmostEqual(ap, META["test_metrics_at_decision_threshold"]["PR_AUC"], places=6)


class TestPredictions(unittest.TestCase):
    def test_typical_case_outputs(self):
        r = predict_one(TYPICAL, MODEL, META)
        self.assertTrue(0.0 <= r["failure_probability"] <= 1.0)
        self.assertIn(r["risk_category"], {"Low", "Medium", "High"})
        self.assertIn(r["predicted_class"], {"Failure", "No failure"})

    def test_matches_raw_pipeline(self):
        r = predict_one(TYPICAL, MODEL, META)
        direct = MODEL.predict_proba(pd.DataFrame([TYPICAL])[INPUT_COLUMNS])[0, 1]
        self.assertAlmostEqual(r["failure_probability"], direct, places=12)

    def test_class_consistent_with_threshold(self):
        for wear in (0, 100, 200, 250):
            r = predict_one(case(**{"Tool wear [min]": wear}), MODEL, META)
            self.assertEqual(r["predicted_failure"], int(r["failure_probability"] >= META["decision_threshold"]))
            self.assertEqual(r["risk_category"], risk_category(r["failure_probability"], META))

    def test_each_machine_type(self):
        for t in ("L", "M", "H"):
            self.assertTrue(0 <= predict_one(case(Type=t), MODEL, META)["failure_probability"] <= 1)

    def test_type_is_case_and_space_insensitive(self):
        a = predict_one(case(Type="m"), MODEL, META)["failure_probability"]
        b = predict_one(case(Type=" M "), MODEL, META)["failure_probability"]
        self.assertEqual(a, b)

    def test_batch_equals_single(self):
        recs = [case(Type="L"), case(**{"Tool wear [min]": 230}), case(**{"Torque [Nm]": 65.0, "Rotational speed [rpm]": 1300})]
        batch = predict_batch(recs, MODEL, META)["failure_probability"].values
        single = [predict_one(r, MODEL, META)["failure_probability"] for r in recs]
        np.testing.assert_allclose(batch, single)

    def test_deterministic(self):
        self.assertEqual(predict_one(TYPICAL, MODEL, META), predict_one(TYPICAL, MODEL, META))

    def test_boundary_values_of_training_range_are_accepted_without_warning(self):
        r = META["observed_ranges"]
        for col in r:
            for v in r[col]:
                out = predict_one(case(**{col: v}), MODEL, META)
                self.assertEqual(out["warnings"], [], f"{col}={v}")

    def test_out_of_range_warns_but_predicts(self):
        out = predict_one(case(**{"Tool wear [min]": 400}), MODEL, META)
        self.assertTrue(out["warnings"])
        self.assertTrue(0 <= out["failure_probability"] <= 1)


class TestInvalidInputs(unittest.TestCase):
    def assert_invalid(self, rec):
        with self.assertRaises(ValueError):
            predict_one(rec, MODEL, META)

    def test_bad_type(self): self.assert_invalid(case(Type="X"))
    def test_negative_torque(self): self.assert_invalid(case(**{"Torque [Nm]": -5}))
    def test_negative_wear(self): self.assert_invalid(case(**{"Tool wear [min]": -1}))
    def test_zero_speed(self): self.assert_invalid(case(**{"Rotational speed [rpm]": 0}))
    def test_zero_kelvin(self): self.assert_invalid(case(**{"Air temperature [K]": 0}))
    def test_text_in_numeric(self): self.assert_invalid(case(**{"Torque [Nm]": "abc"}))
    def test_nan_value(self): self.assert_invalid(case(**{"Torque [Nm]": float("nan")}))
    def test_inf_value(self): self.assert_invalid(case(**{"Torque [Nm]": float("inf")}))
    def test_missing_column(self):
        r = case(); r.pop("Torque [Nm]"); self.assert_invalid(r)

    def test_extra_leakage_columns_are_ignored(self):
        rec = case(); rec.update({"TWF": 1, "HDF": 1, "Machine failure": 1, "UDI": 5})
        clean, _ = validate_input(rec)
        self.assertEqual(list(clean.columns), INPUT_COLUMNS)
        self.assertEqual(predict_one(rec, MODEL, META)["failure_probability"], predict_one(TYPICAL, MODEL, META)["failure_probability"])


if __name__ == "__main__":
    unittest.main()
