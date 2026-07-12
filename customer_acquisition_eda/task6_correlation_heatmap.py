"""
Task 6 - Correlation heatmap
============================

Problem-statement step:
    "Generate a heatmap to illustrate the correlation between different pairs
     of variables."

What this script does
----------------------
1. Builds the numeric correlation matrix on the encoded dataset.
2. Renders a full correlation heatmap and saves it to
   outputs/task6_correlation_heatmap.png.
3. Prints the strongest positive and negative variable pairs so the heatmap is
   easy to interpret.

Run:
    python task6_correlation_heatmap.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import data_prep


def strongest_pairs(corr: pd.DataFrame, n: int = 8):
    """Return the n most strongly correlated (non-self) variable pairs."""
    c = corr.copy()
    # Mask the upper triangle + diagonal so each pair appears once.
    mask = np.triu(np.ones(c.shape, dtype=bool))
    c = c.mask(mask)
    pairs = (
        c.stack()
        .reset_index()
        .rename(columns={"level_0": "Var1", "level_1": "Var2", 0: "corr"})
    )
    pairs["abs"] = pairs["corr"].abs()
    return pairs.sort_values("abs", ascending=False).head(n)


def main():
    out = data_prep.ensure_output_dir()
    print("=" * 70)
    print("TASK 6  |  CORRELATION HEATMAP")
    print("=" * 70)

    # Prefer the fully-encoded dataset from Task 5.
    enc_path = os.path.join(out, "marketing_data_encoded.csv")
    if os.path.exists(enc_path):
        df = pd.read_csv(enc_path)
        print("\n[0] Loaded encoded data from Task 5.")
    else:
        df = data_prep.get_prepared_data()
        print("\n[0] Task-5 output not found; using prepared data.")

    # Focus the heatmap on the analytically meaningful variables (drop ID and
    # the sparse one-hot dummies, which clutter the picture).
    focus = [
        "Age", "Income", "Kids", "Kidhome", "Teenhome", "Recency",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds",
        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "NumWebVisitsMonth",
        "Total_Spending", "Total_Purchases", "Total_Accepted_Cmp",
        "Response", "Complain", "Education_Ordinal",
    ]
    focus = [c for c in focus if c in df.columns]
    corr = df[focus].corr(numeric_only=True)

    # ------------------------------------------------------------------ #
    # Render heatmap
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Correlation heatmap of marketing-campaign variables",
                 fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(f"{out}/task6_correlation_heatmap.png", dpi=110)
    plt.close(fig)
    print("[1] Saved heatmap -> outputs/task6_correlation_heatmap.png")

    # ------------------------------------------------------------------ #
    # Interpret
    # ------------------------------------------------------------------ #
    print("\n[2] Strongest correlated variable pairs:")
    tp = strongest_pairs(corr, n=10)
    for _, r in tp.iterrows():
        print(f"    {r['Var1']:20s} <-> {r['Var2']:20s} : {r['corr']:+.2f}")

    print("\nCONCLUSION")
    print("-" * 70)
    print("* Income, Total_Spending and catalog/store purchases move together")
    print("  strongly (affluent customers buy more across channels).")
    print("* Number of children (Kidhome) correlates negatively with spend and")
    print("  positively with web visits/deal purchases.")


if __name__ == "__main__":
    main()
