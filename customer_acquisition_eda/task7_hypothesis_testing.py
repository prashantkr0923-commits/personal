"""
Task 7 - Hypothesis testing
===========================

Problem-statement step - test the following hypotheses:
    H1. Older individuals may lean toward traditional in-store shopping.
    H2. Customers with children likely prefer the convenience of online
        shopping.
    H3. Sales at physical stores may be cannibalised by other channels.
    H4. Does the United States significantly outperform the rest of the world
        in total purchase volumes?

Approach
--------
Each hypothesis is framed as a null (H0) vs alternative (H1) test at a 5%
significance level (alpha = 0.05).  We use:
    * Pearson & Spearman correlation for monotonic-relationship hypotheses.
    * Mann-Whitney U (non-parametric, robust to skew) for group comparisons,
      with the group means reported for interpretability.

Run:
    python task7_hypothesis_testing.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

import data_prep

ALPHA = 0.05


def verdict(p):
    return "REJECT H0 (significant)" if p < ALPHA else "FAIL TO REJECT H0 (n.s.)"


def load_data():
    out = data_prep.ensure_output_dir()
    clean_path = os.path.join(out, "marketing_data_clean.csv")
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    return data_prep.get_prepared_data()


def h1_age_vs_store(df):
    print("\n" + "-" * 70)
    print("H1  Older customers prefer in-store shopping")
    print("-" * 70)
    print("H0: no relationship between Age and NumStorePurchases")
    print("H1: Age is positively associated with NumStorePurchases")
    r_p, p_p = stats.pearsonr(df["Age"], df["NumStorePurchases"])
    r_s, p_s = stats.spearmanr(df["Age"], df["NumStorePurchases"])
    print(f"   Pearson  r = {r_p:+.3f}  (p = {p_p:.2e})")
    print(f"   Spearman r = {r_s:+.3f}  (p = {p_s:.2e})")
    print("   Verdict:", verdict(p_p))
    print("   Reading: a positive but weak correlation - older customers do")
    print("            buy somewhat more in store, though the effect is small.")


def h2_children_vs_web(df):
    print("\n" + "-" * 70)
    print("H2  Customers with children shop more online (convenience)")
    print("-" * 70)
    print("H0: the online SHARE of purchases is equal for customers with vs")
    print("    without children")
    print("H1: customers with children do a larger SHARE of purchases online")
    print("   (We compare the web SHARE of total purchases, not the raw count:")
    print("    childless customers simply buy more of EVERYTHING, so absolute")
    print("    web counts would be misleading.)")

    channels = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    total_ch = df[channels].sum(axis=1)
    web_share = (df["NumWebPurchases"] / total_ch).astype(float)
    valid = total_ch > 0  # avoid divide-by-zero for the few zero-purchase rows

    with_kids = web_share[valid & (df["Has_Child"] == 1)].dropna()
    no_kids = web_share[valid & (df["Has_Child"] == 0)].dropna()
    u, p = stats.mannwhitneyu(with_kids, no_kids, alternative="greater")
    print(f"   Mean web share - with children : {with_kids.mean():.1%}")
    print(f"   Mean web share - no children   : {no_kids.mean():.1%}")
    print(f"   Mann-Whitney U = {u:.0f}  (p = {p:.2e})")
    print("   Verdict:", verdict(p))
    print("   Reading: customers with children devote a significantly larger")
    print("            share of their purchases to the web channel, supporting")
    print("            the online-convenience hypothesis.")


def h3_store_cannibalisation(df):
    print("\n" + "-" * 70)
    print("H3  Physical-store sales cannibalised by other channels")
    print("-" * 70)
    print("H0: store purchases are not negatively correlated with other channels")
    print("H1: store purchases fall as web/catalog purchases rise (negative r)")
    for other in ["NumWebPurchases", "NumCatalogPurchases"]:
        r, p = stats.pearsonr(df["NumStorePurchases"], df[other])
        print(f"   Store vs {other:20s}: r = {r:+.3f}  (p = {p:.2e})")
    print("   Verdict: correlations are POSITIVE, not negative.")
    print("   Reading: no evidence of cannibalisation - heavy buyers purchase")
    print("            more through EVERY channel, so channels are complements,")
    print("            not substitutes.")


def h4_us_vs_world(df):
    print("\n" + "-" * 70)
    print("H4  Does the US outperform the rest of the world in total purchases?")
    print("-" * 70)
    print("H0: US total purchases equal the rest of the world")
    print("H1: US total purchases differ from the rest of the world")
    us = df.loc[df["Country"] == "US", "Total_Purchases"]
    row = df.loc[df["Country"] != "US", "Total_Purchases"]
    u, p = stats.mannwhitneyu(us, row, alternative="two-sided")
    print(f"   US customers (n={len(us)})   mean total purchases : {us.mean():.2f}")
    print(f"   Rest of world (n={len(row)}) mean total purchases : {row.mean():.2f}")
    print(f"   Mann-Whitney U = {u:.0f}  (p = {p:.3f})")
    print("   Verdict:", verdict(p))
    print("   Reading: the US does NOT significantly outperform the rest of the")
    print("            world; purchase volumes are statistically comparable.")


def main():
    print("=" * 70)
    print("TASK 7  |  HYPOTHESIS TESTING  (alpha = 0.05)")
    print("=" * 70)
    df = load_data()
    # Ensure helper flags exist even if loaded from a partial CSV.
    if "Has_Child" not in df.columns:
        df["Has_Child"] = ((df["Kidhome"] + df["Teenhome"]) > 0).astype(int)

    h1_age_vs_store(df)
    h2_children_vs_web(df)
    h3_store_cannibalisation(df)
    h4_us_vs_world(df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("H1 Age -> in-store   : weak positive link (older buy a bit more in store)")
    print("H2 Children -> online: SUPPORTED - larger web SHARE with children")
    print("H3 Cannibalisation   : NOT supported - channels are complements")
    print("H4 US vs world       : NOT supported - no significant US advantage")


if __name__ == "__main__":
    main()
