"""Dataset inspection (training time) and input validation (inference time)."""
import numpy as np
import pandas as pd

from .features import INPUT_COLUMNS, NUMERIC_RAW, CATEGORICAL, TARGET, FAILURE_MODE_FLAGS, ID_COLUMNS

VALID_TYPES = ["L", "M", "H"]


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a dict of facts about the dataset (used by EDA, the app and the report)."""
    expected = set(INPUT_COLUMNS + [TARGET] + FAILURE_MODE_FLAGS + ID_COLUMNS)
    missing_cols = sorted(expected - set(df.columns))
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
    n_fail = int(df[TARGET].sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df.duplicated(INPUT_COLUMNS).sum()),
        "target_is_binary": bool(set(df[TARGET].unique()) <= {0, 1}),
        "failures": n_fail,
        "non_failures": int(len(df) - n_fail),
        "failure_rate_pct": round(100 * n_fail / len(df), 3),
        "type_counts": df["Type"].value_counts().to_dict(),
        "id_columns_unique": {c: bool(df[c].is_unique) for c in ID_COLUMNS},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def observed_ranges(df: pd.DataFrame) -> dict:
    return {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC_RAW}


def validate_input(records, ranges: dict | None = None):
    """Validate user/model inputs.

    Returns (clean_dataframe, warnings). Raises ValueError for hard errors
    (wrong columns, bad Type, non-numeric, non-finite or physically impossible
    values). Values that are merely OUTSIDE THE TRAINING RANGE only produce a
    warning: the model can still score them, but it is extrapolating.
    """
    df = pd.DataFrame([records]) if isinstance(records, dict) else pd.DataFrame(records).copy()
    absent = [c for c in INPUT_COLUMNS if c not in df.columns]
    if absent:
        raise ValueError(f"Missing required input column(s): {absent}")
    df = df[INPUT_COLUMNS].copy()
    warnings = []

    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    bad_type = ~df["Type"].isin(VALID_TYPES)
    if bad_type.any():
        raise ValueError(f"Type must be one of {VALID_TYPES}; got {df.loc[bad_type, 'Type'].tolist()}")

    for c in NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"'{c}' is missing or not a number.")
        if not np.isfinite(df[c]).all():
            raise ValueError(f"'{c}' must be finite.")

    # Physical impossibilities (hard errors)
    for c in ["Air temperature [K]", "Process temperature [K]"]:
        if (df[c] <= 0).any():
            raise ValueError(f"'{c}' must be above 0 K.")
    for c in ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
        if (df[c] < 0).any():
            raise ValueError(f"'{c}' cannot be negative.")
    if (df["Rotational speed [rpm]"] == 0).any():
        raise ValueError("Rotational speed must be above 0 rpm.")

    if ranges:
        for c in NUMERIC_RAW:
            lo, hi = ranges[c]
            out = (df[c] < lo) | (df[c] > hi)
            if out.any():
                warnings.append(
                    f"{c} = {df.loc[out, c].tolist()} is outside the training range "
                    f"[{lo:g}, {hi:g}]; the prediction is an extrapolation and less reliable."
                )
    return df, warnings
