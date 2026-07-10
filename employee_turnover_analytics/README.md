# Employee Turnover Analytics

Machine Learning course-end project for **Portobello Tech's HR Department** —
predicting employee turnover and recommending retention strategies from
historical evaluation data (14,999 employees).

## What's inside

```
employee_turnover_analytics/
├── data/HR_comma_sep.csv               # source dataset
├── analysis.py                         # standalone pipeline (regenerates everything)
├── build_notebook.py                   # builds the notebook programmatically
├── employee_turnover_analytics.ipynb   # executed notebook (all 7 tasks)
├── employee_turnover_analytics.html    # rendered notebook — open in a browser
├── REPORT.md                           # full written report with findings
├── plots/                              # all generated figures (PNG)
└── outputs/                            # CSV/JSON results
```

## The 7 tasks

1. **Data quality** — missing-value check (0 missing) and imbalance detection.
2. **EDA** — correlation heatmap, distribution plots, project-count bar plot.
3. **Clustering** — K-means (k=3) of employees who left, on satisfaction × evaluation.
4. **Class imbalance** — encode → stratified 80:20 split (`random_state=123`) → SMOTE.
5. **Modeling** — 5-fold CV for Logistic Regression, Random Forest, Gradient Boosting.
6. **Model selection** — ROC/AUC, confusion matrices, Recall-vs-Precision justification.
7. **Retention** — turnover probability → 4 risk zones → tiered action plan.

## Key result

**Random Forest** is the best model: test **AUC ≈ 0.995**, **recall ≈ 0.98** on
the leaver class. **Recall** is the metric of record because missing a real
leaver (false negative) is far costlier than a wasted retention nudge (false
positive).

## Quick start

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
cd employee_turnover_analytics
python analysis.py     # regenerate all plots + outputs
```

Or open `employee_turnover_analytics.html` to read the executed notebook
directly, or re-run it with:

```bash
jupyter nbconvert --to notebook --execute --inplace employee_turnover_analytics.ipynb
```
