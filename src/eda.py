"""
Steps 3-6 (analysis part): exploratory analysis, failure patterns, cleaning checks.
Run from the project root:  python -m src.eda
Writes figures to reports/figures and tables to reports/tables.
Everything here is CROSS-SECTIONAL: the data has no timestamps or machine histories.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_validation import inspect_dataset, observed_ranges
from .features import (TARGET, FAILURE_MODE_FLAGS, NUMERIC_RAW, add_features, ENGINEERED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "machine_failure.csv"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
sns.set_theme(style="whitegrid")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)

    # ---------- Step 2/4: inspection + cleaning checks ----------
    info = inspect_dataset(df)
    info["observed_ranges"] = observed_ranges(df)
    print(json.dumps(info, indent=2))

    df.describe().T.round(3).to_csv(TAB / "numeric_summary.csv")

    rows = []
    for c in NUMERIC_RAW:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        rows.append({"feature": c, "iqr_low": round(lo, 2), "iqr_high": round(hi, 2),
                     "n_outliers": int(mask.sum()),
                     "failure_rate_in_outliers_pct": round(100 * df.loc[mask, TARGET].mean(), 2) if mask.any() else np.nan,
                     "skew": round(df[c].skew(), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "outlier_check.csv", index=False)
    print(out)

    # ---------- Step 3: class balance + failure by category ----------
    counts = df[TARGET].value_counts().rename({0: "No failure", 1: "Failure"})
    plt.figure(figsize=(4.5, 4))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#4C72B0", "#C44E52"], legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({100*v/len(df):.2f}%)", ha="center", va="bottom")
    ax.set_ylabel("Records"); ax.set_title("Target distribution"); ax.set_ylim(0, counts.max() * 1.15)
    savefig("01_class_distribution.png")

    by_type = df.groupby("Type")[TARGET].agg(records="count", failures="sum", failure_rate="mean").loc[["L", "M", "H"]]
    by_type["failure_rate_pct"] = (100 * by_type.pop("failure_rate")).round(2)
    by_type.to_csv(TAB / "failure_by_type.csv")
    plt.figure(figsize=(4.5, 4))
    ax = sns.barplot(x=by_type.index, y=by_type["failure_rate_pct"], hue=by_type.index, legend=False, palette="Blues")
    for i, v in enumerate(by_type["failure_rate_pct"]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("Failure rate (%)"); ax.set_xlabel("Product quality type")
    ax.set_title("Failure rate by machine type")
    savefig("02_failure_by_type.png")

    # ---------- Step 3/5: feature distributions by outcome ----------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), NUMERIC_RAW):
        sns.histplot(data=df, x=c, hue=TARGET, stat="density", common_norm=False, bins=40,
                     palette=PALETTE, element="step", ax=ax)
        ax.set_title(c)
    axes.ravel()[-1].axis("off")
    plt.suptitle("Feature distributions: failure (1) vs no failure (0), density-normalised", y=1.01)
    savefig("03_feature_distributions.png")

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, c in zip(axes, NUMERIC_RAW):
        sns.boxplot(data=df, x=TARGET, y=c, hue=TARGET, palette=PALETTE, legend=False, ax=ax)
        ax.set_title(c, fontsize=9)
    savefig("04_boxplots_by_failure.png")

    summary = df.groupby(TARGET)[NUMERIC_RAW].mean().round(2).T
    summary.columns = ["mean (no failure)", "mean (failure)"]
    summary.to_csv(TAB / "feature_means_by_outcome.csv")

    # ---------- failure rate vs tool wear / torque / rpm bins ----------
    wear_bins = pd.cut(df["Tool wear [min]"], [-1, 50, 100, 150, 200, 220, 240, 260])
    wear_tab = df.groupby(wear_bins, observed=True)[TARGET].agg(records="count", failures="sum", rate="mean")
    wear_tab["failure_rate_pct"] = (100 * wear_tab.pop("rate")).round(2)
    wear_tab.index = wear_tab.index.astype(str)
    wear_tab.to_csv(TAB / "failure_by_tool_wear.csv")
    plt.figure(figsize=(7, 4))
    sns.barplot(x=wear_tab.index, y=wear_tab["failure_rate_pct"], color="#C44E52")
    plt.xticks(rotation=30); plt.ylabel("Failure rate (%)"); plt.xlabel("Tool wear [min] bin")
    plt.title("Failure rate vs tool wear (rises sharply above ~200 min)")
    savefig("05_failure_by_tool_wear.png")

    # ---------- correlation (inputs + target only; flags excluded on purpose) ----------
    eng = add_features(df)
    corr_df = pd.concat([eng.drop(columns=["Type"]), df[TARGET]], axis=1)
    corr = corr_df.corr(method="spearman")
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
    plt.title("Spearman correlation (inputs, engineered features, target)")
    savefig("06_correlation_heatmap.png")
    corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3).to_csv(TAB / "target_correlation.csv")

    # ---------- 2-D views that expose interactions ----------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    s = pd.concat([eng, df[TARGET]], axis=1).sample(frac=1, random_state=0)  # shuffle so failures not hidden
    for ax, (x, y_) in zip(axes, [("Rotational speed [rpm]", "Torque [Nm]"),
                                  ("Rotational speed [rpm]", "temp_diff_K"),
                                  ("Tool wear [min]", "Torque [Nm]")]):
        sns.scatterplot(data=s, x=x, y=y_, hue=TARGET, palette=PALETTE, s=12, alpha=.6, ax=ax)
    axes[0].set_title("Torque vs speed"); axes[1].set_title("Temp difference vs speed"); axes[2].set_title("Torque vs tool wear")
    savefig("07_interactions_scatter.png")

    # ---------- Leakage evidence: failure-mode flags vs target ----------
    flags = df[FAILURE_MODE_FLAGS]
    any_flag = (flags.sum(axis=1) > 0).astype(int)
    leak = pd.crosstab(df[TARGET], any_flag, rownames=["Machine failure"], colnames=["Any mode flag"])
    leak.to_csv(TAB / "flag_vs_target_crosstab.csv")
    flag_tab = pd.DataFrame({"flag_count": flags.sum(),
                             "of_which_machine_failure": flags.mul(df[TARGET], axis=0).sum()})
    flag_tab.to_csv(TAB / "failure_mode_flags.csv")
    print("\nFlag counts:\n", flag_tab, "\nAny-flag vs target:\n", leak)

    # ---------- Step 6: justify engineered features by checking them against the data ----------
    thr = df["Type"].map({"L": 11000, "M": 12000, "H": 13000})
    checks = [
        ("HDF flag", df["HDF"], (eng["temp_diff_K"] < 8.6) & (df["Rotational speed [rpm]"] < 1380),
         "temp_diff < 8.6 K and speed < 1380 rpm"),
        ("PWF flag", df["PWF"], (eng["power_W"] < 3500) | (eng["power_W"] > 9000),
         "power < 3500 W or > 9000 W"),
        ("OSF flag", df["OSF"], eng["wear_torque"] > thr,
         "wear*torque > 11000 (L) / 12000 (M) / 13000 (H)"),
    ]
    rows = []
    for name, flag, rule, desc in checks:
        rows.append({"flag": name, "rule_from_engineered_features": desc,
                     "flag=1 and rule true": int(((flag == 1) & rule).sum()),
                     "flag=1 and rule false": int(((flag == 1) & ~rule).sum()),
                     "flag=0 and rule true": int(((flag == 0) & rule).sum()),
                     "rows where rule true": int(rule.sum()),
                     "failure rate when rule true (%)": round(100 * df.loc[rule, TARGET].mean(), 1)})
    rule_tab = pd.DataFrame(rows)
    rule_tab.to_csv(TAB / "engineered_feature_rule_check.csv", index=False)
    print(rule_tab.T)

    # TWF: failures happen in a random draw inside 200-240 min of wear
    twf = df[df.TWF == 1]["Tool wear [min]"]
    print("TWF wear range:", twf.min(), twf.max())
    info["twf_wear_range"] = [int(twf.min()), int(twf.max())]

    # engineered-feature distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, c in zip(axes, ENGINEERED):
        sns.histplot(data=pd.concat([eng, df[TARGET]], axis=1), x=c, hue=TARGET, stat="density",
                     common_norm=False, bins=40, palette=PALETTE, element="step", ax=ax)
        ax.set_title(c)
    savefig("08_engineered_feature_distributions.png")

    # sequence check: is UDI order a usable time axis?
    info["tool_wear_decreases_in_UDI_order"] = int((df["Tool wear [min]"].diff() < 0).sum())
    info["has_timestamp_column"] = False
    info["no_machine_id_repeats"] = bool(df["Product ID"].is_unique)

    with open(TAB / "dataset_summary.json", "w") as f:
        json.dump(info, f, indent=2)
    print("EDA done. Figures:", len(list(FIG.glob('*.png'))))


if __name__ == "__main__":
    main()
