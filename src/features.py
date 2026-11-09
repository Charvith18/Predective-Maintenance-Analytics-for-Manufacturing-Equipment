"""
Shared constants and feature engineering.

This module is imported by BOTH training and inference, so the saved Joblib
pipeline always applies exactly the same transformations (no train/serve skew).

IMPORTANT: the saved pipeline stores a reference to `src.features.add_features`,
so train.py must be run from the project root as `python -m src.train`, and the
app must import from the same `src` package.
"""
import numpy as np
import pandas as pd

TARGET = "Machine failure"

# Columns that must NEVER be model inputs
FAILURE_MODE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # reveal the target (post-failure info)
ID_COLUMNS = ["UDI", "Product ID"]                         # unique per row, no generalisable signal

# The ONLY columns a user must supply at inference time
CATEGORICAL = ["Type"]
NUMERIC_RAW = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
INPUT_COLUMNS = CATEGORICAL + NUMERIC_RAW

# Engineered, physically motivated features (justified in EDA / report)
ENGINEERED = ["temp_diff_K", "power_W", "wear_torque"]


def select_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """Baseline step: keep only the allowed input columns, in a fixed order."""
    return X[INPUT_COLUMNS].copy()


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw inputs + engineered features.

    temp_diff_K : process temp - air temp (heat-dissipation proxy)
    power_W     : torque [Nm] * angular speed [rad/s]  (mechanical power)
    wear_torque : tool wear [min] * torque [Nm]        (overstrain proxy)
    """
    df = select_inputs(X)
    df["temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60.0
    df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df


def numeric_columns(use_engineered: bool):
    return NUMERIC_RAW + (ENGINEERED if use_engineered else [])
