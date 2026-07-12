"""
Task 1 - Import the data and examine variables for accurate importation
=======================================================================

Problem-statement step:
    "After importing the data, examine variables such as Dt_Customer and
     Income to verify their accurate importation."

What this script does
----------------------
1. Loads the raw CSV exactly as delivered and reports the columns / dtypes so
   the import problems are visible.
2. Highlights the two fields that pandas imports incorrectly:
       * ' Income '  -> the header carries stray spaces and the values are
                        currency strings such as '$84,835.00', so the column
                        is read as text, not a number.
       * 'Dt_Customer' -> read as a plain string instead of a real date.
3. Applies the corrections (via data_prep.fix_data_types) and prints a
   before/after comparison proving the fix worked.

Run:
    python task1_data_import_and_examination.py
"""

import pandas as pd

import data_prep

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)


def main():
    print("=" * 70)
    print("TASK 1  |  DATA IMPORT & VARIABLE EXAMINATION")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load the raw file exactly as it is on disk
    # ------------------------------------------------------------------ #
    raw = data_prep.load_raw_data()
    print("\n[1] Raw dataset loaded")
    print("    Shape (rows, cols):", raw.shape)

    print("\n[2] Raw column names (note the stray spaces around Income):")
    print("   ", list(raw.columns))

    print("\n[3] Raw dtypes as imported by pandas:")
    print(raw.dtypes.to_string())

    # ------------------------------------------------------------------ #
    # 2. Examine the two suspicious variables BEFORE fixing
    # ------------------------------------------------------------------ #
    income_col = " Income "  # the real header, with spaces
    print("\n[4] BEFORE FIX - problem variables")
    print("    ' Income ' sample values :",
          raw[income_col].head(3).tolist(),
          "-> dtype:", raw[income_col].dtype)
    print("    'Dt_Customer' sample values:",
          raw["Dt_Customer"].head(3).tolist(),
          "-> dtype:", raw["Dt_Customer"].dtype)
    print("    Missing Income values     :", int(raw[income_col].isna().sum()))

    # ------------------------------------------------------------------ #
    # 3. Apply the corrections and examine AFTER
    # ------------------------------------------------------------------ #
    fixed = data_prep.fix_data_types(raw)
    print("\n[5] AFTER FIX - corrected variables")
    print("    'Income' dtype      :", fixed["Income"].dtype,
          "| sample:", fixed["Income"].head(3).tolist())
    print("    'Dt_Customer' dtype :", fixed["Dt_Customer"].dtype,
          "| range:", fixed["Dt_Customer"].min().date(),
          "to", fixed["Dt_Customer"].max().date())

    print("\n[6] Numeric summary of the corrected Income field:")
    print(fixed["Income"].describe().round(2).to_string())

    print("\n[7] Full corrected dtypes:")
    print(fixed.dtypes.to_string())

    print("\nCONCLUSION")
    print("-" * 70)
    print("* The income header ' Income ' had surrounding whitespace and the")
    print("  values were currency strings; both are now cleaned to float64.")
    print("* Dt_Customer was text and is now parsed to datetime64 (2012-2014).")
    print("* 24 Income values are missing - handled in Task 2.")


if __name__ == "__main__":
    main()
