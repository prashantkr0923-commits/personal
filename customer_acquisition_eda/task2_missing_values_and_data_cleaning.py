"""
Task 2 - Missing-value imputation and category cleaning
=======================================================

Problem-statement step:
    "There are missing income values for some customers. Conduct missing value
     imputation, considering that customers with similar education and marital
     status tend to have comparable yearly incomes, on average. It may be
     necessary to cleanse the data before proceeding. Specifically, scrutinize
     the categories of education and marital status for data cleaning."

What this script does
----------------------
1. Scrutinises the Education and Marital_Status categories and cleans the
   junk / data-entry values found in Marital_Status ('Alone', 'YOLO',
   'Absurd').
2. Imputes the 24 missing Income values using the median income of each
   (Education, Marital_Status) group - i.e. peers with a comparable profile.
3. Prints before/after evidence that no missing income remains.

Run:
    python task2_missing_values_and_data_cleaning.py
"""

import pandas as pd

import data_prep

pd.set_option("display.width", 120)


def main():
    print("=" * 70)
    print("TASK 2  |  MISSING-VALUE IMPUTATION & DATA CLEANING")
    print("=" * 70)

    # Start from the type-corrected frame produced in Task 1.
    df = data_prep.fix_data_types(data_prep.load_raw_data())

    # ------------------------------------------------------------------ #
    # 1. Scrutinise the categorical variables
    # ------------------------------------------------------------------ #
    print("\n[1] Education categories (all valid education levels):")
    print(df["Education"].value_counts().to_string())

    print("\n[2] Marital_Status categories BEFORE cleaning:")
    print(df["Marital_Status"].value_counts().to_string())
    print("    -> 'Alone', 'YOLO' and 'Absurd' are invalid / junk entries.")

    # ------------------------------------------------------------------ #
    # 2. Clean the categories
    # ------------------------------------------------------------------ #
    df_clean = data_prep.clean_categories(df)
    print("\n[3] Marital_Status categories AFTER cleaning:")
    print(df_clean["Marital_Status"].value_counts().to_string())
    print("    -> 'Alone' treated as 'Single'; the 4 junk rows ('YOLO',")
    print("       'Absurd') recoded to 'Single' as the most plausible value.")

    # ------------------------------------------------------------------ #
    # 3. Missing-income imputation
    # ------------------------------------------------------------------ #
    n_missing = int(df_clean["Income"].isna().sum())
    print(f"\n[4] Missing Income values BEFORE imputation : {n_missing}")

    # Show the group medians that will be used to fill the gaps.
    grp_median = (
        df_clean.groupby(["Education", "Marital_Status"])["Income"]
        .median()
        .round(0)
    )
    print("\n[5] Median income per (Education, Marital_Status) group used as")
    print("    the imputation value (first 10 groups shown):")
    print(grp_median.head(10).to_string())

    df_imputed = data_prep.impute_income(df_clean)
    n_after = int(df_imputed["Income"].isna().sum())
    print(f"\n[6] Missing Income values AFTER imputation  : {n_after}")

    print("\n[7] Income summary after imputation:")
    print(df_imputed["Income"].describe().round(2).to_string())

    print("\nCONCLUSION")
    print("-" * 70)
    print("* Marital_Status cleaned: 8 raw labels -> 5 meaningful categories.")
    print("* All 24 missing incomes filled using education + marital-status")
    print("  peer medians (median chosen because Income is right-skewed with")
    print("  an extreme outlier of 666,666).")


if __name__ == "__main__":
    main()
