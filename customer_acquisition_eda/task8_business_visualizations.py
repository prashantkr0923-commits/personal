"""
Task 8 - Business-question visualizations
=========================================

Problem-statement step - use appropriate visualisation to analyse:
    8.1 Identify the top-performing products and those with the lowest revenue.
    8.2 Examine any correlation between customers' age and the acceptance rate
        of the last campaign (Response).
    8.3 Determine the country with the highest number of customers who accepted
        the last campaign.
    8.4 Investigate any pattern between the number of children at home and total
        expenditure.
    8.5 Analyse the educational background of customers who complained in the
        last two years.

Each analysis prints its finding and saves a chart under outputs/.

Run:
    python task8_business_visualizations.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import data_prep

SPEND_COLS = data_prep.SPEND_COLS
PRODUCT_LABELS = {
    "MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish", "MntSweetProducts": "Sweets",
    "MntGoldProds": "Gold",
}


def load_data():
    out = data_prep.ensure_output_dir()
    clean_path = os.path.join(out, "marketing_data_clean.csv")
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    return data_prep.get_prepared_data()


def q1_products(df, out):
    print("\n[8.1] Top and bottom revenue products")
    revenue = df[SPEND_COLS].sum().rename(index=PRODUCT_LABELS).sort_values()
    print(revenue.to_string())
    print(f"      Top product   : {revenue.idxmax()} ({revenue.max():,.0f})")
    print(f"      Lowest product: {revenue.idxmin()} ({revenue.min():,.0f})")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if v == revenue.min() else
              "#55A868" if v == revenue.max() else "#4C72B0"
              for v in revenue.values]
    ax.bar(revenue.index, revenue.values, color=colors)
    ax.set_title("Total revenue by product category (last 2 years)")
    ax.set_ylabel("Total amount spent")
    for i, v in enumerate(revenue.values):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/task8_1_product_revenue.png", dpi=110)
    plt.close(fig)


def q2_age_vs_response(df, out):
    print("\n[8.2] Age vs acceptance of the last campaign (Response)")
    r, p = stats.pointbiserialr(df["Response"], df["Age"])
    print(f"      Point-biserial corr(Age, Response) = {r:+.3f} (p = {p:.3f})")
    acc = df.loc[df["Response"] == 1, "Age"].mean()
    rej = df.loc[df["Response"] == 0, "Age"].mean()
    print(f"      Mean age accepted={acc:.1f}  vs  not-accepted={rej:.1f}")
    print("      Reading: essentially no correlation - age barely differs")
    print("               between accepters and non-accepters.")

    # Acceptance rate across age bands makes the (non-)pattern visible.
    bands = pd.cut(df["Age"], bins=[0, 30, 40, 50, 60, 70, 120],
                   labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"])
    rate = df.groupby(bands, observed=True)["Response"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rate.index.astype(str), rate.values, color="#4C72B0")
    ax.set_title("Last-campaign acceptance rate by age band")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_xlabel("Age band")
    fig.tight_layout()
    fig.savefig(f"{out}/task8_2_age_vs_response.png", dpi=110)
    plt.close(fig)


def q3_country_accept(df, out):
    print("\n[8.3] Country with most customers who accepted the last campaign")
    accepted = df[df["Response"] == 1]["Country"].value_counts()
    print(accepted.to_string())
    print(f"      Top country: {accepted.idxmax()} ({accepted.max()} customers)")

    fig, ax = plt.subplots(figsize=(9, 5))
    top = accepted.idxmax()
    colors = ["#55A868" if c == top else "#4C72B0" for c in accepted.index]
    ax.bar(accepted.index, accepted.values, color=colors)
    ax.set_title("Customers who accepted the last campaign, by country")
    ax.set_ylabel("Number of accepters")
    fig.tight_layout()
    fig.savefig(f"{out}/task8_3_country_acceptance.png", dpi=110)
    plt.close(fig)


def q4_children_vs_spend(df, out):
    print("\n[8.4] Number of children at home vs total expenditure")
    kids = df["Kidhome"] + df["Teenhome"]
    df = df.assign(_Kids=kids)
    spend_by_kids = df.groupby("_Kids")["Total_Spending"].mean()
    print(spend_by_kids.round(0).to_string())
    print("      Reading: clear downward trend - more children -> lower spend.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(spend_by_kids.index.astype(str), spend_by_kids.values,
           color="#8172B3")
    ax.set_title("Average total expenditure by number of children")
    ax.set_xlabel("Total number of children (Kidhome + Teenhome)")
    ax.set_ylabel("Average total spending")
    fig.tight_layout()
    fig.savefig(f"{out}/task8_4_children_vs_spending.png", dpi=110)
    plt.close(fig)


def q5_complaints_education(df, out):
    print("\n[8.5] Education of customers who complained in the last 2 years")
    complainers = df[df["Complain"] == 1]["Education"].value_counts()
    print(complainers.to_string())
    print(f"      Total complainers: {int(df['Complain'].sum())}")
    if len(complainers):
        print(f"      Most complaints from: {complainers.idxmax()} customers")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(complainers.index, complainers.values, color="#C44E52")
    ax.set_title("Education level of customers who complained (last 2 years)")
    ax.set_ylabel("Number of complainers")
    fig.tight_layout()
    fig.savefig(f"{out}/task8_5_complaints_education.png", dpi=110)
    plt.close(fig)


def main():
    out = data_prep.ensure_output_dir()
    print("=" * 70)
    print("TASK 8  |  BUSINESS-QUESTION VISUALIZATIONS")
    print("=" * 70)
    df = load_data()

    q1_products(df, out)
    q2_age_vs_response(df, out)
    q3_country_accept(df, out)
    q4_children_vs_spend(df, out)
    q5_complaints_education(df, out)

    print("\n[+] Charts saved under outputs/:")
    for f in ["task8_1_product_revenue.png", "task8_2_age_vs_response.png",
              "task8_3_country_acceptance.png",
              "task8_4_children_vs_spending.png",
              "task8_5_complaints_education.png"]:
        print("    -", f)

    print("\nCONCLUSION")
    print("-" * 70)
    print("* Wines & Meat drive revenue; Fruits/Sweets are the smallest lines.")
    print("* Age is not correlated with last-campaign acceptance.")
    print("* Spain (SP) has the most campaign accepters (largest customer base).")
    print("* Spending falls steadily as the number of children rises.")
    print("* Complaints are few and spread across education levels (mostly")
    print("  Graduation, the largest segment).")


if __name__ == "__main__":
    main()
