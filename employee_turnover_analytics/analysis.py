"""
Employee Turnover Analytics
===========================
End-to-end ML pipeline for predicting employee turnover for the HR Department
of Portobello Tech.

Implements the 7 tasks from the project problem statement:
    1. Data quality checks (missing values)
    2. EDA - factors contributing to turnover (heatmap, distributions, bar plot)
    3. K-means clustering of employees who left (satisfaction vs evaluation)
    4. Handle class imbalance with SMOTE (after encoding + stratified split)
    5. 5-fold cross-validated training: Logistic Regression, Random Forest,
       Gradient Boosting (+ classification reports)
    6. Model selection: ROC/AUC curves, confusion matrices, metric justification
    7. Retention strategy: probability zones (Safe/Low/Medium/High risk)

Running this script regenerates every figure in ./plots and every table in
./outputs, and prints a full narrative to stdout.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths & global style
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "HR_comma_sep.csv")
PLOTS = os.path.join(BASE, "plots")
OUT = os.path.join(BASE, "outputs")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 123
sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"stayed": "#2E86AB", "left": "#E4572E"}

results = {}  # collect numeric results for the report


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Task 1 - Data quality checks
# --------------------------------------------------------------------------- #
banner("TASK 1 | Data quality checks")
df = pd.read_csv(DATA)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nColumn dtypes:")
print(df.dtypes)

missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing)
print(f"\nTotal missing values: {int(missing.sum())}")
dupes = int(df.duplicated().sum())
print(f"Duplicate rows: {dupes}")

overall_attrition = df["left"].mean()
print(f"\nOverall attrition rate: {overall_attrition:.4f} "
      f"({df['left'].sum()} of {len(df)} employees left)")

results["task1"] = {
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "total_missing": int(missing.sum()),
    "duplicates": dupes,
    "attrition_rate": float(overall_attrition),
    "n_left": int(df["left"].sum()),
    "n_stayed": int((df["left"] == 0).sum()),
}

# --------------------------------------------------------------------------- #
# Task 2 - EDA
# --------------------------------------------------------------------------- #
banner("TASK 2 | Exploratory Data Analysis")

# 2.1 Correlation heatmap (numerical features)
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("2.1  Correlation Matrix of Numerical Features", pad=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "2_1_correlation_heatmap.png"), dpi=130)
plt.close()

corr_with_left = corr["left"].drop("left").sort_values(key=abs, ascending=False)
print("Correlation of numerical features with `left` (by |value|):")
print(corr_with_left)
results["task2"] = {"corr_with_left": corr_with_left.round(4).to_dict()}

# 2.2 Distribution plots
dist_cols = [
    ("satisfaction_level", "Employee Satisfaction"),
    ("last_evaluation", "Employee Evaluation"),
    ("average_montly_hours", "Average Monthly Hours"),
]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (col, label) in zip(axes, dist_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB", bins=30)
    ax.set_title(label)
    ax.set_xlabel(col)
fig.suptitle("2.2  Distribution Plots", y=1.03, fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "2_2_distributions.png"), dpi=130,
            bbox_inches="tight")
plt.close()

# individual distribution plots too
for col, label in dist_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, color="#2E86AB", bins=30)
    plt.title(f"Distribution of {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"2_2_dist_{col}.png"), dpi=120)
    plt.close()
print("Saved distribution plots for satisfaction / evaluation / monthly hours.")

# 2.3 Bar plot of project count by left
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x="number_project", hue="left",
                   palette=[PALETTE["stayed"], PALETTE["left"]])
ax.set_title("2.3  Project Count of Employees who Stayed vs Left")
ax.set_xlabel("Number of Projects")
ax.set_ylabel("Employee Count")
handles = ax.get_legend().legend_handles if ax.get_legend() else []
ax.legend(title="left", labels=["Stayed (0)", "Left (1)"])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "2_3_project_count_bar.png"), dpi=130)
plt.close()

proj = pd.crosstab(df["number_project"], df["left"])
proj.columns = ["stayed", "left"]
proj["attrition_rate"] = proj["left"] / (proj["left"] + proj["stayed"])
print("\nProject count vs attrition:")
print(proj)
results["task2"]["project_attrition"] = proj.round(4).reset_index().to_dict("records")

# --------------------------------------------------------------------------- #
# Task 3 - K-means clustering of employees who left
# --------------------------------------------------------------------------- #
banner("TASK 3 | K-means clustering of employees who left")
left_df = df[df["left"] == 1][["satisfaction_level", "last_evaluation"]].copy()
km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
left_df["cluster"] = km.fit_predict(left_df)

centers = pd.DataFrame(km.cluster_centers_,
                       columns=["satisfaction_level", "last_evaluation"])
print("Cluster centers (employees who left):")
print(centers)
print("\nCluster sizes:")
print(left_df["cluster"].value_counts().sort_index())

plt.figure(figsize=(10, 7))
sns.scatterplot(data=left_df, x="satisfaction_level", y="last_evaluation",
                hue="cluster", palette="Set1", alpha=0.6, s=40)
plt.scatter(centers["satisfaction_level"], centers["last_evaluation"],
            c="black", s=260, marker="X", label="centroids")
plt.title("3.2  K-means Clusters of Employees who Left (k=3)")
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.legend(title="cluster")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "3_2_kmeans_clusters.png"), dpi=130)
plt.close()

# label clusters semantically by their centroids
labels = {}
for i, r in centers.iterrows():
    s, e = r["satisfaction_level"], r["last_evaluation"]
    if s < 0.5 and e >= 0.7:
        labels[i] = "Burned-out high performers (low satisfaction, high eval)"
    elif s >= 0.5 and e >= 0.7:
        labels[i] = "Frustrated high performers (decent satisfaction, high eval)"
    else:
        labels[i] = "Disengaged / under-evaluated (low satisfaction, low eval)"
print("\nSemantic cluster interpretation:")
for i in sorted(labels):
    print(f"  Cluster {i}: {labels[i]}")

left_df.to_csv(os.path.join(OUT, "left_employee_clusters.csv"), index=False)
results["task3"] = {
    "centers": centers.round(4).to_dict("records"),
    "sizes": left_df["cluster"].value_counts().sort_index().to_dict(),
    "labels": labels,
}

# --------------------------------------------------------------------------- #
# Task 4 - Handle class imbalance with SMOTE
# --------------------------------------------------------------------------- #
banner("TASK 4 | Pre-processing, stratified split & SMOTE")

# 4.1 encode categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Categorical columns: {cat_cols}")
categorical = df[cat_cols]
numeric = df.drop(columns=cat_cols)
cat_dummies = pd.get_dummies(categorical, drop_first=False)
data = pd.concat([numeric, cat_dummies], axis=1)

X = data.drop(columns=["left"])
y = data["left"]
print(f"Feature matrix after encoding: {X.shape}")

# 4.2 stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train class balance BEFORE SMOTE:")
print(y_train.value_counts())

# 4.3 SMOTE upsample train only
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nTrain class balance AFTER SMOTE:")
print(y_train_res.value_counts())

results["task4"] = {
    "n_features": int(X.shape[1]),
    "train_before": y_train.value_counts().to_dict(),
    "train_after": y_train_res.value_counts().to_dict(),
    "test_size": int(X_test.shape[0]),
}

# --------------------------------------------------------------------------- #
# Task 5 - 5-fold CV model training
# --------------------------------------------------------------------------- #
banner("TASK 5 | 5-fold cross-validated model training")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000,
                                               random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200,
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv_reports = {}
for name, model in models.items():
    y_cv_pred = cross_val_predict(model, X_train_res, y_train_res, cv=cv)
    rep = classification_report(y_train_res, y_cv_pred, output_dict=True)
    cv_reports[name] = rep
    print(f"\n--- {name} | 5-fold CV classification report (on SMOTE train) ---")
    print(classification_report(y_train_res, y_cv_pred,
                                target_names=["stayed", "left"]))

# classification-report comparison figure (f1 by model)
rep_rows = []
for name, rep in cv_reports.items():
    rep_rows.append({
        "model": name,
        "accuracy": rep["accuracy"],
        "precision_left": rep["1"]["precision"],
        "recall_left": rep["1"]["recall"],
        "f1_left": rep["1"]["f1-score"],
    })
rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(os.path.join(OUT, "cv_classification_reports.csv"), index=False)
print("\nCV summary (class = left):")
print(rep_df.round(4))

fig, ax = plt.subplots(figsize=(11, 6))
metrics_plot = rep_df.set_index("model")[
    ["precision_left", "recall_left", "f1_left", "accuracy"]]
metrics_plot.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title("5.  5-fold CV metrics by model (positive class = left)")
ax.set_ylabel("Score")
ax.set_ylim(0.8, 1.0)
ax.legend(loc="lower right")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "5_cv_metrics_comparison.png"), dpi=130)
plt.close()

results["task5"] = rep_df.round(4).to_dict("records")

# --------------------------------------------------------------------------- #
# Task 6 - Best model selection: ROC/AUC + confusion matrix
# --------------------------------------------------------------------------- #
banner("TASK 6 | ROC/AUC, confusion matrices & metric justification")

# fit each model on full SMOTE train, evaluate on untouched test set
fitted, test_metrics = {}, {}
plt.figure(figsize=(9, 8))
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    fitted[name] = model
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    cm = confusion_matrix(y_test, pred)
    rep = classification_report(y_test, pred, output_dict=True)
    test_metrics[name] = {
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "recall_left": rep["1"]["recall"],
        "precision_left": rep["1"]["precision"],
        "f1_left": rep["1"]["f1-score"],
        "accuracy": rep["accuracy"],
    }
    print(f"{name}: test AUC={auc:.4f}  recall(left)={rep['1']['recall']:.4f}  "
          f"precision(left)={rep['1']['precision']:.4f}")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("6.1  ROC Curves (test set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "6_1_roc_curves.png"), dpi=130)
plt.close()

# 6.2 confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, model) in zip(axes, fitted.items()):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["stayed", "left"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAUC={test_metrics[name]['auc']:.3f}")
fig.suptitle("6.2  Confusion Matrices (test set)", y=1.03, fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "6_2_confusion_matrices.png"), dpi=130,
            bbox_inches="tight")
plt.close()

best_model_name = max(test_metrics, key=lambda n: (test_metrics[n]["auc"],
                                                    test_metrics[n]["recall_left"]))
best_model = fitted[best_model_name]
print(f"\nBEST MODEL: {best_model_name} "
      f"(AUC={test_metrics[best_model_name]['auc']:.4f}, "
      f"recall={test_metrics[best_model_name]['recall_left']:.4f})")
results["task6"] = {"test_metrics": test_metrics, "best_model": best_model_name}

# --------------------------------------------------------------------------- #
# Task 7 - Retention strategy: probability zones
# --------------------------------------------------------------------------- #
banner("TASK 7 | Retention strategy zones (best model)")

proba_test = best_model.predict_proba(X_test)[:, 1]


def zone(p):
    if p < 0.20:
        return "Safe Zone (Green)"
    elif p < 0.60:
        return "Low-Risk Zone (Yellow)"
    elif p < 0.90:
        return "Medium-Risk Zone (Orange)"
    return "High-Risk Zone (Red)"


zones = pd.Series([zone(p) for p in proba_test], index=X_test.index)
zone_order = ["Safe Zone (Green)", "Low-Risk Zone (Yellow)",
              "Medium-Risk Zone (Orange)", "High-Risk Zone (Red)"]

zone_out = X_test.copy()
zone_out["actual_left"] = y_test.values
zone_out["turnover_probability"] = proba_test
zone_out["risk_zone"] = zones.values
zone_out.sort_values("turnover_probability", ascending=False).to_csv(
    os.path.join(OUT, "employee_risk_zones.csv"), index=False)

zone_counts = zones.value_counts().reindex(zone_order).fillna(0).astype(int)
print("Employees per risk zone (test set):")
print(zone_counts)

plt.figure(figsize=(10, 6))
colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]
ax = sns.barplot(x=zone_counts.index, y=zone_counts.values, palette=colors)
for i, v in enumerate(zone_counts.values):
    ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
ax.set_title(f"7.  Employees by Turnover-Risk Zone ({best_model_name})")
ax.set_ylabel("Employee Count")
ax.set_xlabel("")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "7_risk_zones.png"), dpi=130)
plt.close()

results["task7"] = {"zone_counts": zone_counts.to_dict(),
                    "best_model": best_model_name}

# --------------------------------------------------------------------------- #
# Persist all numeric results
# --------------------------------------------------------------------------- #
with open(os.path.join(OUT, "results_summary.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

banner("DONE - all plots in ./plots, tables in ./outputs")
print(json.dumps({k: (v if not isinstance(v, dict) else "…")
                  for k, v in results.items()}, indent=2, default=str))
