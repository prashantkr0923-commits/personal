# Crime Analysis — Exploratory Data Analysis & Cluster Analysis

Course-End Project (Crime Analysis). This repository reproduces the five
analytical tasks from the problem statement programmatically in Python and
produces a set of **downloadable outputs** — charts, summary tables, an Excel
workbook, and a single self-contained HTML report.

## Tasks covered

| # | Task | What it answers |
|---|------|-----------------|
| 1 | **Overall Crime Statistics** | Count & types of crimes, geographic distribution, most common incidents, live-feed snapshot (YTD total + most recent incident). |
| 2 | **Time-Period Analysis** | Distribution by day-of-week and hour; % of incidents by time block (Morning/Afternoon/Evening/Night); day×hour heatmap. |
| 3 | **Trend Analysis** | Monthly trajectory, same-month year-over-year comparison, evolution of the crime-type mix. |
| 4 | **Comparative Analysis** | Arrest vs. no-arrest split; arrest rate by crime type and city; severity composition per crime type. |
| 5 | **Cluster Analysis** | K-means segmentation of incidents (elbow + silhouette, PCA view, cluster fingerprints) and hierarchical clustering of cities by crime-type profile. |

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn scipy openpyxl
python crime_analysis.py
```

By default the script reads the dataset from the path in `CRIME_CSV`
(overridable via environment variable). All outputs are written to `outputs/`.

## Outputs

```
outputs/
├── Crime_Analysis_Report.html      ← self-contained report (all charts + findings)
├── Crime_Analysis_Summary.xlsx     ← every summary table as a worksheet
├── crime_data_enriched.csv         ← source data + engineered fields
├── charts/                         ← 19 PNG charts (colour-blind-safe palette)
└── tables/                         ← 21 CSV summary tables
```

Open **`outputs/Crime_Analysis_Report.html`** in a browser for the full,
narrated write-up with key findings per task.

## Dataset

1,000 incidents across 8 cities (Austin, Dallas, Houston, San Antonio, Tulsa,
Oklahoma City, Kansas City, Little Rock), 8 crime types, spanning
2023-06-11 → 2025-06-10. The data is clean (no nulls, no duplicates). Each row
is a single incident (`Number of Crimes` is always 1).

### Note on the schema

The generic variable dictionary in the problem statement (Case Number, Block,
UCR, **Domestic**, Beat, District, Ward, …) describes a Chicago-style crime feed
and does **not** match the columns actually shipped in the CSV. In particular
there is **no `Domestic` field**, so Task 4's requested *domestic-share*
breakdown is adapted to a **severity-composition** breakdown per crime type,
which is the closest available category dimension. This substitution is stated
explicitly in the report.

## Method notes (Task 5)

- Incidents are clustered on **circumstance** features only — hour, day-of-week,
  severity, latitude, longitude. Arrest outcome is deliberately **excluded** from
  the inputs (it is near-collinear with `Number of Arrests` and would otherwise
  trivially split the data into arrest/no-arrest) and instead **profiled as an
  outcome** of each segment.
- `k` is chosen by silhouette score; the resulting segments are largely
  **regional** (geography dominates the feature space), and clearance rate is
  found to be roughly uniform across regions — a useful negative result.
- Cities are additionally clustered with **Ward-linkage hierarchical
  clustering** on their crime-type share profile (dendrogram + 3-group cut).
