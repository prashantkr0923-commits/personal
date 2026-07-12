"""
Task 3 - Feature engineering (derived variables)
================================================

Problem-statement steps:
    "Create variables to represent the total number of children, age, and
     total spending."
    "Derive the total purchases from the number of transactions across the
     three channels."

What this script does
----------------------
Creates the derived analytical variables on top of the cleaned + imputed data:

    Kids            = Kidhome + Teenhome            (total children)
    Age             = 2015 - Year_Birth             (reference year 2015)
    Total_Spending  = sum of the 6 Mnt* spend cols  (Product pillar)
    Total_Purchases = Web + Catalog + Store         (three sales channels)

It also builds a couple of helper flags (Total_Accepted_Cmp, Has_Child, Is_US)
reused by the hypothesis-testing script.

Run:
    python task3_feature_engineering.py
"""

import pandas as pd

import data_prep

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)


def main():
    print("=" * 70)
    print("TASK 3  |  FEATURE ENGINEERING")
    print("=" * 70)

    # Cleaned + imputed data (Tasks 1-2), before feature engineering.
    df = data_prep.impute_income(
        data_prep.clean_categories(
            data_prep.fix_data_types(data_prep.load_raw_data())
        )
    )

    # Apply Task-3 feature engineering.
    df = data_prep.engineer_features(df)

    print("\n[1] New derived columns created:")
    new_cols = ["Kids", "Age", "Total_Spending", "Total_Purchases",
                "Total_Accepted_Cmp", "Has_Child", "Is_US"]
    print("   ", new_cols)

    print("\n[2] Definition recap:")
    print("    Kids            = Kidhome + Teenhome")
    print("    Age             = 2015 - Year_Birth")
    print("    Total_Spending  = MntWines + MntFruits + MntMeatProducts")
    print("                      + MntFishProducts + MntSweetProducts + MntGoldProds")
    print("    Total_Purchases = NumWebPurchases + NumCatalogPurchases")
    print("                      + NumStorePurchases")

    print("\n[3] Sample of engineered features:")
    print(df[["Year_Birth", "Age", "Kidhome", "Teenhome", "Kids",
              "Total_Spending", "Total_Purchases"]].head(8).to_string(index=False))

    print("\n[4] Summary statistics of the new numeric features:")
    print(df[["Age", "Kids", "Total_Spending", "Total_Purchases"]]
          .describe().round(2).to_string())

    print("\n[5] Sanity checks / notes:")
    print("    Max Age =", int(df["Age"].max()),
          "(implausible birth years 1893/1899/1900 -> outliers for Task 4).")
    print("    Kids distribution:", df["Kids"].value_counts().sort_index().to_dict())

    print("\nCONCLUSION")
    print("-" * 70)
    print("* 4 core engineered variables (Kids, Age, Total_Spending,")
    print("  Total_Purchases) plus helper flags are now available for the")
    print("  distribution, correlation and hypothesis-testing tasks.")


if __name__ == "__main__":
    main()
