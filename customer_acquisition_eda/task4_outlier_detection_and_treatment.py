"""
Task 4 - Distributions, box plots, histograms and outlier treatment
===================================================================

Problem-statement step:
    "Generate box plots and histograms to gain insights into the distributions
     and identify outliers. Implement outlier treatment as needed."

What this script does
----------------------
1. Draws histograms and box plots for the key numeric variables and saves them
   to outputs/ (task4_histograms.png, task4_boxplots_before.png).
2. Detects outliers with the IQR rule and reports the count per variable.
3. Treats outliers:
       * Removes physically impossible ages (Age > 100, i.e. Year_Birth
         1893/1899/1900).
       * Caps the remaining heavy-tailed variables at their IQR fences
         (winsorising) so extreme values no longer distort later analysis.
4. Saves post-treatment box plots (task4_boxplots_after.png) and writes the
   cleaned dataset to outputs/marketing_data_clean.csv for reuse.

Run:
    python task4_outlier_detection_and_treatment.py
"""

import matplotlib
matplotlib.use("Agg")  # headless backend - render straight to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_prep

# Variables we inspect for outliers / distribution shape.
NUMERIC_VARS = [
    "Age", "Income", "Total_Spending", "Total_Purchases",
    "Recency", "NumWebVisitsMonth",
]

# Continuous variables we winsorise (cap) rather than delete rows.
CAP_VARS = ["Income", "Total_Spending", "Total_Purchases", "NumWebVisitsMonth"]


def iqr_bounds(series: pd.Series, k: float = 1.5):
    """Return the lower/upper IQR fences for a numeric series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def main():
    out = data_prep.ensure_output_dir()
    print("=" * 70)
    print("TASK 4  |  DISTRIBUTIONS, OUTLIERS & TREATMENT")
    print("=" * 70)

    df = data_prep.get_prepared_data()

    # ------------------------------------------------------------------ #
    # 1. Histograms
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.ravel(), NUMERIC_VARS):
        ax.hist(df[col], bins=40, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Histogram - {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    fig.suptitle("Distributions of key numeric variables", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{out}/task4_histograms.png", dpi=110)
    plt.close(fig)
    print("\n[1] Saved histograms          -> outputs/task4_histograms.png")

    # ------------------------------------------------------------------ #
    # 2. Box plots BEFORE treatment
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.ravel(), NUMERIC_VARS):
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor="#DD8452"))
        ax.set_title(f"Box plot - {col}")
        ax.set_ylabel(col)
    fig.suptitle("Box plots BEFORE outlier treatment", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{out}/task4_boxplots_before.png", dpi=110)
    plt.close(fig)
    print("[2] Saved box plots (before)  -> outputs/task4_boxplots_before.png")

    # ------------------------------------------------------------------ #
    # 3. Detect outliers with the IQR rule
    # ------------------------------------------------------------------ #
    print("\n[3] Outlier counts (IQR 1.5x rule):")
    for col in NUMERIC_VARS:
        lo, hi = iqr_bounds(df[col])
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        print(f"    {col:18s}: {n:4d} outliers  (fences {lo:.0f} .. {hi:.0f})")

    # ------------------------------------------------------------------ #
    # 4. Treat outliers
    # ------------------------------------------------------------------ #
    before_rows = len(df)

    # 4a. Drop impossible ages (data-entry errors in Year_Birth).
    df = df[df["Age"] <= 100].copy()
    print(f"\n[4] Removed {before_rows - len(df)} rows with Age > 100 "
          f"(impossible birth years).")

    # 4b. Winsorise the heavy-tailed continuous variables at their IQR fences.
    for col in CAP_VARS:
        lo, hi = iqr_bounds(df[col])
        df[col] = df[col].clip(lower=lo, upper=hi)
    print(f"[5] Winsorised (capped at IQR fences): {CAP_VARS}")

    # ------------------------------------------------------------------ #
    # 5. Box plots AFTER treatment
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.ravel(), NUMERIC_VARS):
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor="#55A868"))
        ax.set_title(f"Box plot - {col}")
        ax.set_ylabel(col)
    fig.suptitle("Box plots AFTER outlier treatment", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{out}/task4_boxplots_after.png", dpi=110)
    plt.close(fig)
    print("[6] Saved box plots (after)   -> outputs/task4_boxplots_after.png")

    # ------------------------------------------------------------------ #
    # 6. Persist the cleaned dataset for downstream reuse
    # ------------------------------------------------------------------ #
    clean_path = f"{out}/marketing_data_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"[7] Saved treated dataset     -> {clean_path}  (rows: {len(df)})")

    print("\nCONCLUSION")
    print("-" * 70)
    print("* Income, Total_Spending and Age were the most outlier-prone.")
    print("* 3 impossible-age rows removed; continuous variables winsorised.")
    print("* Cleaned data saved for the encoding / correlation / hypothesis")
    print("  scripts to reuse.")


if __name__ == "__main__":
    main()
