# Simulation Screenshots — index

All figures are produced by the scripts in `src/` and saved to
`outputs/figures/`. Each is a screenshot of a simulation/analysis step.

| # | File | What it shows | Produced by |
|---|------|---------------|-------------|
| 01 | `01_solidity_findings.png` | Solidity coverage (134/240), distribution, and solidity vs blade count | `02_solidity_analysis.py` |
| 02 | `02_target_distributions.png` | Univariate distributions of C_T, C_P, η | `03_eda_visualizations.py` |
| 03 | `03_performance_curves.png` | Bivariate: C_T, C_P, η vs advance ratio J (the classic propeller curves) | `03_eda_visualizations.py` |
| 04 | `04_bivariate_drivers.png` | C_T & C_P vs solidity; efficiency by blade count | `03_eda_visualizations.py` |
| 05 | `05_correlation_heatmap.png` | Correlation heatmap of all numeric variables | `03_eda_visualizations.py` |
| 06 | `06_model_comparison_r2.png` | R² of the three Gradient-Boosting missing-value strategies | `04_ml_models.py` |
| 07 | `07_pred_vs_actual.png` | Predicted vs actual (variant B) on held-out 3/4-blade props | `04_ml_models.py` |
| 08 | `08_feature_importance.png` | Gradient Boosting feature importances per target | `04_ml_models.py` |
| 09 | `09_sql_summary.png` | Week-1 SQL answers summary | `05_run_sql.py` |
| 10 | `10_dashboard.png` | Tableau-style executive storytelling dashboard | `06_tableau_dashboard.py` |
