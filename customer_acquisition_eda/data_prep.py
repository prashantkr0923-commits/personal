"""
data_prep.py
============
Shared data-preparation utilities for the Marketing Campaigns EDA project.

This module encapsulates the reusable cleaning / feature-engineering logic that
Tasks 1, 2 and 3 build up, so that the later analysis scripts (Tasks 4-8) can
all start from an identical, reproducible, analysis-ready DataFrame.

Every task script imports the helpers it needs from here.  The functions are
written as a small, ordered pipeline:

    load_raw_data()        ->  raw DataFrame exactly as it sits on disk
    fix_data_types()       ->  Task 1  : correct Income / Dt_Customer imports
    clean_categories()     ->  Task 2a : tidy Education & Marital_Status
    impute_income()        ->  Task 2b : fill missing Income by group
    engineer_features()    ->  Task 3  : Age, children, spend, purchases ...
    get_prepared_data()    ->  convenience wrapper running the whole chain

Author : Data Science Course-End Project
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The CSV lives next to this module, so we resolve the path relative to __file__
# and the analysis scripts can be run from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "marketing_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# The dataset was compiled shortly after the last enrolment (June 2014).  We use
# 2015 as the fixed reference year so that "Age" is reproducible regardless of
# when the script is run.
REFERENCE_YEAR = 2015

# The six product-spend columns ("Product" pillar of the marketing mix).
SPEND_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

# The three purchase *channels* ("Place" pillar).  NumDealsPurchases is a
# promotion mechanic, not a channel, so it is deliberately excluded here.
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]


def ensure_output_dir():
    """Create the outputs/ folder if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Task 1 - import & type correction
# ---------------------------------------------------------------------------
def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as delivered (no cleaning applied).

    Note that the header for the income field is literally ' Income ' (with
    surrounding spaces) and the values are currency strings such as
    '$84,835.00', so the column arrives as an object/string dtype.
    """
    return pd.read_csv(path)


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the two columns that pandas imports with the wrong dtype.

    * Column names are stripped of stray whitespace (' Income ' -> 'Income').
    * 'Income' is converted from a currency string to a float.
    * 'Dt_Customer' is parsed from the m/d/yy text format to datetime64.
    """
    df = df.copy()

    # 1) Trim whitespace around every column label.
    df.columns = df.columns.str.strip()

    # 2) Income: strip '$' and thousands separators, then cast to float.
    #    The raw column is text (e.g. '$84,835.00'); only convert if it is not
    #    already numeric so the function is safe to call more than once.
    if not pd.api.types.is_numeric_dtype(df["Income"]):
        df["Income"] = (
            df["Income"]
            .astype("string")
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
            .astype(float)
        )

    # 3) Dt_Customer: the file uses month/day/two-digit-year (e.g. 6/16/14).
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%m/%d/%y")

    return df


# ---------------------------------------------------------------------------
# Task 2a - category cleaning
# ---------------------------------------------------------------------------
def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy the Education and Marital_Status categorical variables.

    Marital_Status contains three non-sensical / data-entry values -
    'Alone', 'YOLO' and 'Absurd'.  'Alone' clearly means a single person, and
    the two junk labels ('YOLO', 'Absurd' - 4 rows total) are re-coded to
    'Single' as the most plausible category rather than discarded.

    Education is already clean (Basic, 2n Cycle, Graduation, Master, PhD); the
    label '2n Cycle' is standardised to the clearer 'Master' equivalent is
    intentionally *avoided* because we need the five distinct ordinal levels
    for encoding later, so we simply leave Education untouched.
    """
    df = df.copy()

    marital_map = {
        "Alone": "Single",
        "YOLO": "Single",
        "Absurd": "Single",
    }
    df["Marital_Status"] = df["Marital_Status"].replace(marital_map)

    return df


# ---------------------------------------------------------------------------
# Task 2b - missing-value imputation
# ---------------------------------------------------------------------------
def impute_income(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the 24 missing Income values.

    The brief states that customers with a similar education and marital status
    tend to have comparable incomes, so we fill each gap with the *median*
    income of its (Education, Marital_Status) group.  The median is preferred
    over the mean because Income is heavily right-skewed and contains an extreme
    outlier (666,666), which would inflate a group mean.
    """
    df = df.copy()

    df["Income"] = df.groupby(["Education", "Marital_Status"])["Income"].transform(
        lambda s: s.fillna(s.median())
    )

    # Safety net: if any group were entirely missing, fall back to global median.
    df["Income"] = df["Income"].fillna(df["Income"].median())

    return df


# ---------------------------------------------------------------------------
# Task 3 - feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived analytical variables required by the project.

    * Kids            : total children  = Kidhome + Teenhome
    * Age             : REFERENCE_YEAR - Year_Birth
    * Total_Spending  : sum of the six Mnt* product-spend columns
    * Total_Purchases : sum of the three channel purchase counts
    * Total_Accepted_Cmp : how many of the 5 campaigns the customer accepted
    * Has_Child       : boolean helper used in hypothesis testing
    * Is_US           : boolean helper used in hypothesis testing
    """
    df = df.copy()

    df["Kids"] = df["Kidhome"] + df["Teenhome"]
    df["Age"] = REFERENCE_YEAR - df["Year_Birth"]
    df["Total_Spending"] = df[SPEND_COLS].sum(axis=1)
    df["Total_Purchases"] = df[CHANNEL_COLS].sum(axis=1)
    df["Total_Accepted_Cmp"] = df[
        ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    ].sum(axis=1)
    df["Has_Child"] = (df["Kids"] > 0).astype(int)
    df["Is_US"] = (df["Country"] == "US").astype(int)

    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def get_prepared_data(with_features: bool = True) -> pd.DataFrame:
    """Run the full clean -> impute -> (optionally) engineer pipeline.

    Parameters
    ----------
    with_features : bool
        If True (default) the returned frame includes the Task-3 derived
        columns.  Pass False when a script only needs the cleaned raw fields.
    """
    df = load_raw_data()
    df = fix_data_types(df)
    df = clean_categories(df)
    df = impute_income(df)
    if with_features:
        df = engineer_features(df)
    return df


if __name__ == "__main__":
    # Quick self-test so `python data_prep.py` confirms the pipeline runs.
    data = get_prepared_data()
    print("Prepared dataset shape :", data.shape)
    print("Remaining null Income  :", int(data["Income"].isna().sum()))
    print("New engineered columns :",
          ["Kids", "Age", "Total_Spending", "Total_Purchases",
           "Total_Accepted_Cmp", "Has_Child", "Is_US"])
