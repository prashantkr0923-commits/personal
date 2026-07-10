"""Build the Employee Turnover Analytics Jupyter notebook programmatically."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Employee Turnover Analytics
### Portobello Tech — HR Department | Machine Learning Course-End Project

**Objective:** Build ML programs to predict employee turnover and recommend
retention strategies.

This notebook implements all seven required tasks:

1. Data quality checks (missing values)
2. EDA — factors contributing to turnover (heatmap, distribution plots, project bar plot)
3. K-means clustering of employees who left (satisfaction vs. evaluation)
4. Handle class imbalance with **SMOTE** (encoding → stratified 80:20 split → upsample)
5. **5-fold cross-validated** training — Logistic Regression, Random Forest, Gradient Boosting
6. Best-model selection — ROC/AUC, confusion matrices, Recall-vs-Precision justification
7. Retention strategy — categorize employees into risk zones and recommend actions
""")

code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_predict)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 123
pd.set_option("display.float_format", lambda x: f"{x:.4f}")""")

# ---- Task 1 ----
md("""## Task 1 — Data Quality Checks

Load the dataset and check for missing values (and duplicates as a bonus quality check).""")
code("""df = pd.read_csv("data/HR_comma_sep.csv")
print("Shape:", df.shape)
df.head()""")
code("""print("Missing values per column:")
print(df.isnull().sum())
print("\\nTotal missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))""")
md("""**Result:** The dataset has **0 missing values** across all 14,999 rows and 10 columns —
no imputation is required. There are 3,008 fully duplicated rows; these are legitimate
employees with identical profiles and are **kept** so the class distribution and model
signal are preserved (dropping them is optional and does not change the conclusions).""")
code("""print(f"Overall attrition rate: {df['left'].mean():.4f}")
df['left'].value_counts()""")
md("""The target `left` is **imbalanced**: ~76% stayed vs ~24% left — motivating SMOTE in Task 4.""")

# ---- Task 2 ----
md("""## Task 2 — EDA: What factors contribute most to turnover?

### 2.1 Correlation heatmap of numerical features""")
code("""num_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(11, 9))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

print("Correlation with `left` (sorted by magnitude):")
print(num_df.corr()["left"].drop("left").sort_values(key=abs, ascending=False))""")
md("""**Inference:** `satisfaction_level` has the strongest (negative) correlation with
leaving (**-0.39**) — unhappy employees leave. `Work_accident` (-0.15) and
`time_spend_company` (+0.14) follow. Interestingly `last_evaluation` and
`number_project` show almost no *linear* correlation, yet the plots below reveal strong
*non-linear* relationships — which is why tree ensembles outperform logistic regression.""")

md("""### 2.2 Distribution plots — Satisfaction, Evaluation, Monthly Hours""")
code("""dist_cols = [("satisfaction_level", "Employee Satisfaction"),
             ("last_evaluation", "Employee Evaluation"),
             ("average_montly_hours", "Average Monthly Hours")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (col, label) in zip(axes, dist_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB", bins=30)
    ax.set_title(label)
plt.tight_layout()
plt.show()""")
md("""**Inference:** All three distributions are **bimodal**. Satisfaction has a spike near
0.1 (a distinct unhappy group). Evaluation and monthly hours each show two peaks — a
lower-engagement cluster and an over-worked high-performer cluster — hinting at two very
different reasons employees leave.""")

md("""### 2.3 Project count — employees who stayed vs left""")
code("""plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x="number_project", hue="left",
                   palette=["#2E86AB", "#E4572E"])
ax.set_title("Project Count of Employees who Stayed vs Left")
ax.legend(title="left", labels=["Stayed (0)", "Left (1)"])
plt.tight_layout()
plt.show()

proj = pd.crosstab(df["number_project"], df["left"])
proj.columns = ["stayed", "left"]
proj["attrition_rate"] = proj["left"] / proj.sum(axis=1)
proj""")
md("""**Inference:** Attrition is **U-shaped** in project count:
- **2 projects → 66% leave** (under-utilised / disengaged).
- **3–4 projects → lowest attrition (~2–9%)** — the healthy "sweet spot".
- **6 projects → 56% leave, and 7 projects → 100% leave** (over-loaded / burnt-out).

Both extremes — too few and too many projects — drive people out.""")

# ---- Task 3 ----
md("""## Task 3 — K-means clustering of employees who left

### 3.1–3.2 Cluster the leavers on satisfaction & evaluation into 3 groups""")
code("""left_df = df[df["left"] == 1][["satisfaction_level", "last_evaluation"]].copy()
km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
left_df["cluster"] = km.fit_predict(left_df[["satisfaction_level", "last_evaluation"]])

centers = pd.DataFrame(km.cluster_centers_,
                       columns=["satisfaction_level", "last_evaluation"])
print("Cluster centers:"); print(centers)
print("\\nCluster sizes:"); print(left_df["cluster"].value_counts().sort_index())""")
code("""plt.figure(figsize=(10, 7))
sns.scatterplot(data=left_df, x="satisfaction_level", y="last_evaluation",
                hue="cluster", palette="Set1", alpha=0.6, s=40)
plt.scatter(centers["satisfaction_level"], centers["last_evaluation"],
            c="black", s=260, marker="X", label="centroids")
plt.title("K-means Clusters of Employees who Left (k=3)")
plt.legend(title="cluster")
plt.tight_layout()
plt.show()""")
md("""### 3.3 Interpretation of the three leaver clusters

| Cluster | Satisfaction | Evaluation | Profile |
|---|---|---|---|
| **Burned-out high performers** | ~0.11 (very low) | ~0.87 (very high) | Excellent employees who were over-worked and disengaged — the most costly to lose. |
| **Disengaged / under-evaluated** | ~0.41 (low-mid) | ~0.52 (low) | Mediocre engagement and ratings — a mix of poor fit and low motivation. |
| **Frustrated high performers** | ~0.81 (high) | ~0.91 (very high) | Satisfied *and* high-performing, yet still left — likely for better external offers, pay, or lack of promotion. |

Each cluster needs a **different** retention lever (workload, engagement, career growth).""")

# ---- Task 4 ----
md("""## Task 4 — Handle class imbalance with SMOTE

### 4.1 Encode categorical columns (get_dummies)""")
code("""cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

categorical = df[cat_cols]
numeric = df.drop(columns=cat_cols)
cat_dummies = pd.get_dummies(categorical, drop_first=False)
data = pd.concat([numeric, cat_dummies], axis=1)

X = data.drop(columns=["left"])
y = data["left"]
print("Encoded feature matrix:", X.shape)""")
md("""### 4.2 Stratified 80:20 train/test split (`random_state=123`)""")
code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train balance BEFORE SMOTE:"); print(y_train.value_counts())""")
md("""### 4.3 Upsample the training set with SMOTE (imblearn)""")
code("""smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Train balance AFTER SMOTE:"); print(y_train_res.value_counts())""")
md("""SMOTE is applied **only to the training set** (never the test set) so the test
evaluation reflects the true, imbalanced population. The two classes are now balanced
at 9,142 each.""")

# ---- Task 5 ----
md("""## Task 5 — 5-fold cross-validated model training

For each model we run **stratified 5-fold CV** on the SMOTE-balanced training data and
print the classification report from out-of-fold predictions.""")
code("""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
cv_reports = {}
for name, model in models.items():
    y_cv_pred = cross_val_predict(model, X_train_res, y_train_res, cv=cv)
    cv_reports[name] = classification_report(y_train_res, y_cv_pred, output_dict=True)
    print(f"===== {name} — 5-fold CV classification report =====")
    print(classification_report(y_train_res, y_cv_pred,
                                target_names=["stayed", "left"]))""")
code("""rep_df = pd.DataFrame([{
    "model": n, "accuracy": r["accuracy"],
    "precision_left": r["1"]["precision"], "recall_left": r["1"]["recall"],
    "f1_left": r["1"]["f1-score"]} for n, r in cv_reports.items()])
display(rep_df)

ax = rep_df.set_index("model")[["precision_left","recall_left","f1_left","accuracy"]] \\
        .plot(kind="bar", figsize=(11,6), colormap="viridis")
ax.set_ylim(0.8, 1.0); ax.set_title("5-fold CV metrics by model (class = left)")
plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.show()""")
md("""**Result:** Random Forest leads on every CV metric (F1 ≈ 0.98), Gradient Boosting is
close behind (F1 ≈ 0.96), and Logistic Regression trails (F1 ≈ 0.81) because the
turnover signal is strongly non-linear.""")

# ---- Task 6 ----
md("""## Task 6 — Best model: ROC/AUC, confusion matrices & metric choice

Each model is fit on the full SMOTE training set and evaluated on the **untouched test
set**.

### 6.1 ROC curves & AUC""")
code("""fitted, test_metrics = {}, {}
plt.figure(figsize=(9, 8))
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    fitted[name] = model
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    rep = classification_report(y_test, model.predict(X_test), output_dict=True)
    test_metrics[name] = {"auc": auc, "recall": rep["1"]["recall"],
                          "precision": rep["1"]["precision"]}
plt.plot([0,1],[0,1],"k--",lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves (test set)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.show()
pd.DataFrame(test_metrics).T""")
md("""### 6.2 Confusion matrices""")
code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, fitted.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["stayed","left"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\\nAUC={test_metrics[name]['auc']:.3f}")
plt.tight_layout(); plt.show()""")
md("""### 6.3 Which metric matters — Recall or Precision?

**Recall (of the "left" class) is the priority metric here.**

- A **false negative** = an employee who *will* leave but the model says "safe". HR does
  nothing, the person quits, and the company eats the full cost of backfilling
  (hiring, onboarding, lost productivity, knowledge loss). This is the **expensive** error.
- A **false positive** = an employee flagged at-risk who would actually have stayed. The
  only cost is a low-price retention gesture (a check-in, a small perk). This is **cheap**.

Because missing a real leaver is far costlier than a wasted retention nudge, we optimise
to **catch as many true leavers as possible → maximise Recall**. AUC is used as the
overall ranking metric since it is threshold-independent.

**Best model: Random Forest** — highest test AUC (**≈ 0.995**) and highest recall on the
leaver class (**≈ 0.98**).""")

# ---- Task 7 ----
md("""## Task 7 — Retention strategy by risk zone

### 7.1 Predict turnover probability on the test set with the best model""")
code("""best_model = fitted["Random Forest"]
proba_test = best_model.predict_proba(X_test)[:, 1]

def zone(p):
    if p < 0.20:  return "Safe Zone (Green)"
    if p < 0.60:  return "Low-Risk Zone (Yellow)"
    if p < 0.90:  return "Medium-Risk Zone (Orange)"
    return "High-Risk Zone (Red)"

zones = pd.Series([zone(p) for p in proba_test], index=X_test.index)
order = ["Safe Zone (Green)","Low-Risk Zone (Yellow)",
         "Medium-Risk Zone (Orange)","High-Risk Zone (Red)"]
zone_counts = zones.value_counts().reindex(order).fillna(0).astype(int)
print(zone_counts)

plt.figure(figsize=(10,6))
ax = sns.barplot(x=zone_counts.index, y=zone_counts.values,
                 palette=["#2ECC71","#F1C40F","#E67E22","#E74C3C"])
for i,v in enumerate(zone_counts.values): ax.text(i, v+2, str(v), ha="center", fontweight="bold")
ax.set_title("Employees by Turnover-Risk Zone (Random Forest)"); ax.set_ylabel("Count")
plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()""")
md("""### 7.2 Recommended retention strategies per zone

| Zone | Probability | Strategy |
|---|---|---|
| 🟢 **Safe** | < 20% | No intervention needed. Keep normal engagement, recognition and career-development cycles. Use them as mentors / culture anchors. |
| 🟡 **Low-Risk** | 20–60% | Preventive. Regular 1:1 check-ins, ensure workload sits in the healthy 3–4 project range, provide learning opportunities and timely feedback. |
| 🟠 **Medium-Risk** | 60–90% | Active. Manager conversation to surface concerns; review compensation and promotion eligibility; rebalance workload; set a concrete 3-month growth plan. |
| 🔴 **High-Risk** | > 90% | Urgent, personalised retention. Immediate skip-level/HR conversation, targeted counter-offer or role change, address burnout (cap projects/hours), fast-track promotion where deserved. Prioritise the *burned-out high performers* first. |

**Tie the zones back to the clusters (Task 3):** high-risk *burned-out high performers*
need workload relief and recognition; *frustrated high performers* need pay/promotion;
*disengaged* employees need role-fit and engagement work.""")

md("""## Summary

- **Data**: 14,999 employees, 0 missing values, 24% attrition (imbalanced).
- **Top drivers**: low satisfaction, extreme project loads (2 or 6–7), long tenure, no recent promotion.
- **Leaver segments**: burned-out high performers, disengaged staff, frustrated high performers.
- **Best model**: **Random Forest** — test AUC ≈ 0.995, recall (left) ≈ 0.98.
- **Metric of record**: **Recall** on the leaver class (missing a leaver is the costly error).
- **Action**: score employees → assign risk zones → apply the tiered retention playbook above.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
with open("employee_turnover_analytics.ipynb", "w") as f:
    nbf.write(nb, f)
print("Notebook written with", len(cells), "cells.")
