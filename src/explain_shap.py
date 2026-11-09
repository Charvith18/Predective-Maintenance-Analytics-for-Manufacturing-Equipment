"""
OPTIONAL SHAP analysis (requires `pip install shap`).   python -m src.explain_shap
NOTE: written for tree-based final models (Random Forest / HistGradientBoosting / Decision Tree).
It could NOT be executed in the environment where this project was built (no shap installed),
so treat it as untested until you run it once.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .features import INPUT_COLUMNS, TARGET
from .predict import load_artifacts

ROOT = Path(__file__).resolve().parents[1]

model, meta = load_artifacts()
df = pd.read_csv(ROOT / "data" / "machine_failure.csv")
X_tr, X_te, y_tr, y_te = train_test_split(df[INPUT_COLUMNS], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=meta["random_seed"])

# Use one calibrated sub-model's underlying pipeline if the saved model is a CalibratedClassifierCV
base = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
prep, clf = base[:-1], base[-1]
Xt = prep.transform(X_te)
names = [n.replace("num__", "").replace("cat__", "") for n in base.named_steps["prep"].get_feature_names_out()]
Xt = pd.DataFrame(Xt.toarray() if hasattr(Xt, "toarray") else Xt, columns=names)

sv = shap.TreeExplainer(clf).shap_values(Xt)
sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
shap.summary_plot(sv, Xt, show=False)
plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "17_shap_summary.png", dpi=130, bbox_inches="tight")
print("Saved reports/figures/17_shap_summary.png")
