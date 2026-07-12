"""
Task 5 - Ordinal and one-hot encoding of categorical variables
==============================================================

Problem-statement step:
    "Apply ordinal and one-hot encoding based on the various types of
     categorical variables."

Encoding strategy
-----------------
* Education has a natural ranking, so it is ORDINAL-encoded:
      Basic (0) < 2n Cycle (1) < Graduation (2) < Master (3) < PhD (4)
* Marital_Status and Country are nominal (no inherent order), so they are
  ONE-HOT encoded with pandas.get_dummies (drop_first=True to avoid the dummy
  trap).

The script prints the resulting columns and writes the fully-encoded,
model-ready dataset to outputs/marketing_data_encoded.csv.

Run:
    python task5_categorical_encoding.py
"""

import os

import pandas as pd

import data_prep

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 60)

# Natural ordering for the ordinal Education variable.
EDUCATION_ORDER = {
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4,
}


def main():
    out = data_prep.ensure_output_dir()
    print("=" * 70)
    print("TASK 5  |  ORDINAL & ONE-HOT ENCODING")
    print("=" * 70)

    # Reuse the outlier-treated dataset from Task 4 if it exists; otherwise
    # rebuild the prepared data so this script runs stand-alone.
    clean_path = os.path.join(out, "marketing_data_clean.csv")
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        print("\n[0] Loaded outlier-treated data from Task 4.")
    else:
        df = data_prep.get_prepared_data()
        print("\n[0] Task-4 output not found; rebuilt prepared data.")

    # ------------------------------------------------------------------ #
    # 1. Ordinal encoding of Education
    # ------------------------------------------------------------------ #
    df["Education_Ordinal"] = df["Education"].map(EDUCATION_ORDER)
    print("\n[1] ORDINAL encoding of Education:")
    print("    Mapping:", EDUCATION_ORDER)
    print(df[["Education", "Education_Ordinal"]]
          .drop_duplicates()
          .sort_values("Education_Ordinal")
          .to_string(index=False))

    # ------------------------------------------------------------------ #
    # 2. One-hot encoding of the nominal variables
    # ------------------------------------------------------------------ #
    nominal = ["Marital_Status", "Country"]
    df_encoded = pd.get_dummies(
        df, columns=nominal, drop_first=True, dtype=int
    )
    new_dummies = [c for c in df_encoded.columns
                   if c.startswith(("Marital_Status_", "Country_"))]
    print("\n[2] ONE-HOT encoding of Marital_Status and Country:")
    print("    Created", len(new_dummies), "dummy columns:")
    print("   ", new_dummies)

    # The original Education text column is now redundant; keep the ordinal one.
    df_encoded = df_encoded.drop(columns=["Education"])

    print("\n[3] Final encoded dataset shape:", df_encoded.shape)
    print("[4] Sample of encoded columns:")
    preview = ["Education_Ordinal"] + new_dummies[:4]
    print(df_encoded[preview].head(6).to_string(index=False))

    # ------------------------------------------------------------------ #
    # 3. Persist the encoded dataset
    # ------------------------------------------------------------------ #
    enc_path = os.path.join(out, "marketing_data_encoded.csv")
    df_encoded.to_csv(enc_path, index=False)
    print(f"\n[5] Saved encoded dataset -> {enc_path}")

    print("\nCONCLUSION")
    print("-" * 70)
    print("* Education -> single ordinal column preserving its natural order.")
    print("* Marital_Status & Country -> one-hot dummies (first level dropped).")
    print("* Dataset is now fully numeric and ready for correlation/modelling.")


if __name__ == "__main__":
    main()
