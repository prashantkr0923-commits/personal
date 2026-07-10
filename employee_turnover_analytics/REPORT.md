# Employee Turnover Analytics — Project Report
**Portobello Tech · HR Department · Machine Learning Course-End Project**

---

## Executive Summary

Portobello Tech's HR department needs to predict which employees are likely to
leave so it can intervene before they resign. Using historical evaluation data
for **14,999 employees**, this project builds an end-to-end ML pipeline that
(a) identifies the drivers of turnover, (b) segments the employees who left,
(c) trains and compares three classifiers under 5-fold cross-validation, and
(d) turns the best model's probabilities into an actionable, tiered retention
playbook.

**Headline results**

| Item | Finding |
|---|---|
| Data quality | 0 missing values; 24% overall attrition (imbalanced) |
| Strongest driver | Low `satisfaction_level` (corr −0.39 with leaving) |
| Best model | **Random Forest** — test **AUC ≈ 0.995**, **recall(left) ≈ 0.98** |
| Priority metric | **Recall** on the "left" class (missing a leaver is the costly error) |
| Action | Score employees → 4 risk zones → tiered retention strategy |

---

## Task 1 — Data Quality Checks

- Dataset: **14,999 rows × 10 columns**.
- **Missing values: 0** in every column → no imputation required.
- Duplicate rows: 3,008 (kept — they are valid identical employee profiles and
  removing them is optional; it does not change the conclusions).
- Target `left`: **3,571 left (23.8%)** vs 11,428 stayed → **class imbalance**,
  addressed with SMOTE in Task 4.

## Task 2 — EDA: Drivers of Turnover

**2.1 Correlation with `left` (by magnitude):**

| Feature | Corr with `left` |
|---|---|
| satisfaction_level | **−0.39** |
| Work_accident | −0.15 |
| time_spend_company | +0.14 |
| average_montly_hours | +0.07 |
| promotion_last_5years | −0.06 |
| number_project | +0.02 |
| last_evaluation | +0.01 |

Satisfaction is by far the strongest linear signal. Evaluation and project
count look uncorrelated *linearly* but are highly predictive *non-linearly*
(see 2.2/2.3) — the reason tree ensembles beat logistic regression.

**2.2 Distributions** (`satisfaction_level`, `last_evaluation`,
`average_montly_hours`) are all **bimodal**, revealing two distinct employee
populations — a low-engagement group and an over-worked high-performer group.

**2.3 Project count vs attrition (U-shaped):**

| # Projects | Attrition rate |
|---|---|
| 2 | **65.6%** (under-utilised) |
| 3 | 1.8% |
| 4 | 9.4% |
| 5 | 22.2% |
| 6 | 55.8% |
| 7 | **100%** (burnt-out) |

> **Inference:** Both extremes drive people out. The healthy "sweet spot" is
> **3–4 projects**; anyone on 2 or on 6–7 projects is a flight risk.

## Task 3 — Clustering the Employees Who Left (K-means, k=3)

Clustering the leavers on `satisfaction_level` × `last_evaluation` yields three
clean, well-separated groups:

| Cluster | Satisfaction | Evaluation | Interpretation |
|---|---|---|---|
| Burned-out high performers | ~0.11 | ~0.87 | Great employees, over-worked & unhappy — most costly to lose |
| Disengaged / under-evaluated | ~0.41 | ~0.52 | Low engagement and ratings — poor fit / low motivation |
| Frustrated high performers | ~0.81 | ~0.91 | Happy *and* excellent yet still left — better offers / no promotion |

Each segment needs a **different** retention lever (workload, engagement,
career growth respectively).

## Task 4 — Class Imbalance with SMOTE

1. **Encoding:** categorical columns (`sales`/department, `salary`) separated
   from numerics, one-hot encoded with `get_dummies()`, then recombined →
   **18 features**.
2. **Stratified 80:20 split** with `random_state=123` (11,999 train / 3,000 test).
3. **SMOTE** applied to the **training set only** (imblearn) → balanced to
   **9,142 vs 9,142**. The test set is left untouched so evaluation reflects the
   real population.

## Task 5 — 5-fold Cross-Validated Training

Stratified 5-fold CV on the SMOTE-balanced training data (positive class = `left`):

| Model | Accuracy | Precision (left) | Recall (left) | F1 (left) |
|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.801 | 0.821 | 0.811 |
| **Random Forest** | **0.985** | **0.995** | **0.975** | **0.985** |
| Gradient Boosting | 0.963 | 0.976 | 0.949 | 0.962 |

Random Forest dominates; Logistic Regression trails because turnover is highly
non-linear.

## Task 6 — Best Model: ROC/AUC, Confusion Matrix, Metric Choice

Evaluated on the untouched test set:

| Model | Test AUC | Recall (left) | Precision (left) |
|---|---|---|---|
| Logistic Regression | 0.821 | 0.719 | 0.533 |
| **Random Forest** | **0.995** | **0.978** | **0.979** |
| Gradient Boosting | 0.986 | 0.930 | 0.918 |

**6.3 Recall vs Precision — use Recall.**
A **false negative** (a real leaver the model calls "safe") is expensive: HR
does nothing and the company pays the full cost of backfilling the role. A
**false positive** (flagging someone who would have stayed) only costs a cheap
retention gesture. Since missing a leaver is far costlier, we **maximise recall
of the "left" class**, with AUC as the threshold-independent ranking metric.

> **Best model: Random Forest** (AUC ≈ 0.995, recall ≈ 0.98).

## Task 7 — Retention Strategy by Risk Zone

Using the Random Forest turnover probabilities on the test set:

| Zone | Probability | Employees (test) | Retention Strategy |
|---|---|---|---|
| 🟢 Safe | < 20% | 2,182 | No action; keep normal engagement/recognition; use as mentors. |
| 🟡 Low-Risk | 20–60% | 111 | Preventive: regular 1:1s, keep workload at 3–4 projects, learning opportunities. |
| 🟠 Medium-Risk | 60–90% | 61 | Active: manager conversation, review pay/promotion, rebalance workload, 3-month plan. |
| 🔴 High-Risk | > 90% | 646 | Urgent, personalised: skip-level/HR talk, counter-offer or role change, address burnout, fast-track promotion. |

Link zones back to clusters: high-risk **burned-out high performers** → workload
relief + recognition; **frustrated high performers** → pay/promotion;
**disengaged** → role-fit + engagement.

---

## Deliverables

| File | Contents |
|---|---|
| `employee_turnover_analytics.ipynb` | Executed notebook — all 7 tasks with code, plots, narrative |
| `employee_turnover_analytics.html` | Rendered notebook (open in any browser) |
| `analysis.py` | Standalone script that regenerates every plot/table |
| `REPORT.md` | This report |
| `plots/` | All figures (heatmap, distributions, clusters, ROC, confusion matrices, risk zones) |
| `outputs/left_employee_clusters.csv` | Cluster label for each leaver |
| `outputs/cv_classification_reports.csv` | 5-fold CV metrics per model |
| `outputs/employee_risk_zones.csv` | Per-employee turnover probability + risk zone |
| `outputs/results_summary.json` | All numeric results in machine-readable form |

## How to Reproduce

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
cd employee_turnover_analytics
python analysis.py                 # regenerate plots + outputs
# or run the notebook end-to-end:
jupyter nbconvert --to notebook --execute --inplace employee_turnover_analytics.ipynb
```
