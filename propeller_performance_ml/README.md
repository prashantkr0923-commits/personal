# UAV Propeller Performance — Data Science Capstone

A data-driven framework that predicts propeller **thrust coefficient (C_T)**,
**power coefficient (C_P)** and **efficiency (η)** from blade geometry and
operating inputs, using standalone and **ensemble (Gradient Boosting)** machine
learning. Built for the small-UAV / urban-air-mobility context.

**Start here:** [`report/REPORT.md`](report/REPORT.md) — the full write-up
(understanding, approach, methods, results, conclusions).

## What's inside
```
propeller_performance_ml/
├── data/
│   ├── raw/            Experiment_vol1-3.csv, Geom_vol1-2.csv  (source data)
│   └── processed/      appended + engineered tables, Tableau extract
├── src/
│   ├── config.py                 shared paths + to_identifier() helper
│   ├── 01_data_preparation.py    append versions, rename, radius/chord distributions
│   ├── 02_solidity_analysis.py   numpy.trapz blade area, disc area, solidity, collation
│   ├── 03_eda_visualizations.py  univariate/bivariate analysis + correlation heatmap
│   ├── 04_ml_models.py           Gradient Boosting; 3 missing-value variants; comparison
│   ├── 05_run_sql.py             Week-1 SQL answers (SQLite)
│   ├── 06_tableau_dashboard.py   Tableau extract + rendered storytelling dashboard
│   ├── sql/week1_queries.sql     the four Week-1 SQL queries
│   └── run_all.py                run the whole pipeline
├── outputs/
│   ├── figures/        10 figures (the "simulation screenshots")
│   └── tables/         solidity table, model comparison, SQL result sets
└── report/
    ├── REPORT.md                 main write-up
    ├── tableau_dashboard_spec.md how to build the dashboard in Tableau
    ├── SCREENSHOTS.md            index of every figure
    └── propeller_diagram.png     chord/radius reference (from the brief)
```

## Reproduce
```bash
pip install -r requirements.txt
cd src && python run_all.py
```

## Deliverables mapped to the brief
| Deliverable | Where |
|-------------|-------|
| **1. Write-up** (understanding, approach, methods, solutions) | `report/REPORT.md` |
| **2. Screenshots of simulations** | `outputs/figures/*.png` (indexed in `report/SCREENSHOTS.md`) |
| **3. Source code** | `src/**` and `src/sql/week1_queries.sql` |

## Headline results
- **SQL:** 42 propellers exceed 12 % thrust coefficient in experiment 1; all
  240 propellers hit η ≤ 0 at high advance ratio.
- **Solidity:** computable for 134/240 propellers (106 lack geometry); rises
  with blade count (0.089 → 0.137 → 0.163 for 2/3/4 blades).
- **ML (Gradient Boosting, train on 2-blade → test on 3-/4-blade):** C_T R² up
  to **0.66**, C_P **0.49**, η **0.58**. Imputation beats row-deletion; a
  **solidity-free model generalises best** across blade counts.
