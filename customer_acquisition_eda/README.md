# Marketing Campaigns – Customer Acquisition EDA

Course-end project: **Applied Data Science with Python**.

As a data scientist, the goal is to run exploratory data analysis and
hypothesis testing on a marketing-campaign dataset (structured around the
marketing mix – *People, Product, Place, Promotion*) to better understand the
factors that influence customer acquisition.

The eight problem-statement tasks are each delivered as a **self-contained,
runnable Python file**. Shared cleaning / feature-engineering logic lives in
`data_prep.py` so every script starts from an identical, reproducible dataset.

## Files

| File | Task |
|------|------|
| `data_prep.py` | Reusable pipeline: load → fix dtypes → clean categories → impute income → engineer features |
| `task1_data_import_and_examination.py` | **Task 1** – Import data; verify `Income` & `Dt_Customer` imported correctly |
| `task2_missing_values_and_data_cleaning.py` | **Task 2** – Impute missing income by education + marital-status peers; clean junk categories |
| `task3_feature_engineering.py` | **Task 3** – Create `Kids`, `Age`, `Total_Spending`, `Total_Purchases` |
| `task4_outlier_detection_and_treatment.py` | **Task 4** – Histograms + box plots; IQR outlier detection & treatment |
| `task5_categorical_encoding.py` | **Task 5** – Ordinal encode `Education`; one-hot encode `Marital_Status` & `Country` |
| `task6_correlation_heatmap.py` | **Task 6** – Correlation heatmap of all variable pairs |
| `task7_hypothesis_testing.py` | **Task 7** – Test the four business hypotheses |
| `task8_business_visualizations.py` | **Task 8** – Five business-question visualisations |
| `run_all.py` | Runs Tasks 1–8 end to end |
| `marketing_data.csv` | Raw input data |
| `Data_Dictionary.xlsx` | Variable descriptions |
| `outputs/` | Generated charts (`.png`) and cleaned/encoded datasets (`.csv`) |

## How to run

```bash
pip install -r requirements.txt

# run a single task
python task1_data_import_and_examination.py

# or run everything
python run_all.py
```

Each script prints its findings to the console and (where relevant) writes
charts to `outputs/`. Run the tasks in order the first time: Task 4 saves the
outlier-treated dataset that Tasks 5–8 reuse.

## Key findings

**Data preparation**
- `Income` arrived as a currency string under a mis-spaced header `' Income '`; `Dt_Customer` arrived as text. Both corrected (float / datetime).
- 24 missing incomes imputed with the **median** income of each `(Education, Marital_Status)` group (median chosen because income is right-skewed with a 666,666 outlier).
- `Marital_Status` cleaned from 8 raw labels (incl. junk `YOLO`, `Absurd`, `Alone`) to 5 meaningful categories.
- 3 impossible-age rows removed (birth years 1893/1899/1900); heavy-tailed variables winsorised at IQR fences.

**Hypothesis testing** (α = 0.05)

| # | Hypothesis | Result |
|---|------------|--------|
| H1 | Older customers prefer in-store shopping | **Weak support** – significant but small positive correlation (r ≈ 0.14) |
| H2 | Customers with children shop more online | **Supported** – larger *share* of purchases via web (35% vs 27%) |
| H3 | Physical stores cannibalised by other channels | **Not supported** – channels are complements (positive correlations) |
| H4 | US outperforms the rest of the world | **Not supported** – no significant difference in total purchases |

**Business insights**
- **Wines** and **Meat** dominate revenue; **Fruits** and **Sweets** are the smallest lines.
- Age is essentially **uncorrelated** with last-campaign acceptance.
- **Spain (SP)** has by far the most customers who accepted the last campaign (largest customer base).
- Total spending **falls** as the number of children rises.
- Complaints are rare (~20) and concentrated in the largest segment (`Graduation`).

> Reference year for `Age` is fixed at **2015** (the data was compiled just after the last enrolment in June 2014) so results are reproducible.
