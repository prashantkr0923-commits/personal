"""
04_ml_models.py  -  Week 2 (Machine Learning)
==============================================
Tasks addressed
---------------
1. Detect missing values and perform missing-value treatment.
2. Predict propeller performance (thrust coefficient C_T, power coefficient
   C_P, efficiency eta) from blade geometry + operational inputs, using the
   GRADIENT BOOSTING technique.
3. TRAIN on the 2-blade propellers, then EVALUATE on the "other" propellers
   (the held-out 3-blade and 4-blade propellers).
4. Build THREE model variants and compare them:
       (A) Without missing-value imputation  -> complete-case (drop rows whose
           solidity is missing).
       (B) With missing-value imputation      -> median-impute solidity.
       (C) Without solidity                    -> drop the solidity feature.

Where do the missing values come from?
   The experiment coefficients themselves are complete, but 106 of 240
   propellers have NO blade-geometry rows, so their SOLIDITY cannot be
   computed. After collation, ~48% of experiment rows have a missing solidity.
   Solidity is therefore the feature that drives the missing-value treatment.

Outputs:
   outputs/tables/model_comparison.csv
   outputs/figures/06_model_comparison_r2.png
   outputs/figures/07_pred_vs_actual.png
   outputs/figures/08_feature_importance.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import PROCESSED, FIGURES, TABLES

TARGETS = {
    "thrust_coefficient_output": "C_T",
    "power_coefficient_output": "C_P",
    "efficiency_output": "eta",
}
# Base operational + geometric features. number_of_blades is intentionally
# EXCLUDED: the model trains only on 2-blade props (zero variance there) and
# must generalise to 3/4-blade props, so it cannot use blade count as an input.
BASE_FEATURES = ["propeller_diameter", "propeller_pitch",
                 "advanced_ratio_input", "rpm_rotation_input"]
SOLIDITY = "solidity"


def treat_efficiency_outliers(df):
    """Efficiency is only physically meaningful in the working state. Past the
    zero-thrust advance ratio the propeller windmills/brakes and eta = J*CT/CP
    explodes toward large negatives as CP -> 0. We clip the target to a
    physically sensible band [-1, 0.9]; CT and CP are left untouched (clean)."""
    out = df.copy()
    n_bad = int(((out["efficiency_output"] < -1) |
                 (out["efficiency_output"] > 0.9)).sum())
    out["efficiency_output"] = out["efficiency_output"].clip(-1.0, 0.9)
    print(f"  efficiency outlier treatment: clipped {n_bad} extreme rows to [-1, 0.9]")
    return out


def fit_eval(train, test, features, target, impute=False):
    """Fit a GradientBoostingRegressor and score it on the test frame.

    Returns (metrics dict, fitted model, imputer-or-None, y_true, y_pred)."""
    Xtr, ytr = train[features].copy(), train[target].values
    Xte, yte = test[features].copy(), test[target].values

    imputer = None
    if impute and SOLIDITY in features:
        imputer = SimpleImputer(strategy="median")
        Xtr[features] = imputer.fit_transform(Xtr[features])
        Xte[features] = imputer.transform(Xte[features])

    model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {
        "R2": r2_score(yte, pred),
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "MAE": mean_absolute_error(yte, pred),
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }
    return metrics, model, imputer, yte, pred


def main():
    df = pd.read_csv(os.path.join(PROCESSED, "experiment_with_solidity.csv"))

    # ---- 1. Missing-value audit ------------------------------------------
    print("STEP 1  Missing-value audit (collated experiment + solidity)")
    miss = df.isna().sum()
    print(miss[miss > 0].to_string())
    print(f"  rows with missing solidity: {int(df[SOLIDITY].isna().sum())} "
          f"/ {len(df)}  ({100*df[SOLIDITY].isna().mean():.1f}%)")

    df = treat_efficiency_outliers(df)

    # ---- Train / test split by blade count -------------------------------
    train_all = df[df["number_of_blades"] == 2].copy()
    test_all = df[df["number_of_blades"] != 2].copy()   # 3- and 4-blade props
    print(f"\nSTEP 3  Split -> train (2 blades): {len(train_all)} rows / "
          f"{train_all['propeller_name'].nunique()} props | "
          f"test (3&4 blades): {len(test_all)} rows / "
          f"{test_all['propeller_name'].nunique()} props")

    # Common test subset with KNOWN solidity => fair head-to-head for all 3
    test_known = test_all.dropna(subset=[SOLIDITY]).copy()
    print(f"  fair-comparison test subset (solidity known): {len(test_known)} rows")

    # ---- 2 & 4. Three model variants -------------------------------------
    variants = {
        "A. Without imputation (complete-case)":
            dict(features=BASE_FEATURES + [SOLIDITY], impute=False,
                 drop_missing_train=True),
        "B. With imputation (median solidity)":
            dict(features=BASE_FEATURES + [SOLIDITY], impute=True,
                 drop_missing_train=False),
        "C. Without solidity":
            dict(features=BASE_FEATURES, impute=False,
                 drop_missing_train=False),
    }

    rows = []
    stash = {}   # (variant, target) -> (y_true, y_pred, model, features)
    for vname, cfg in variants.items():
        feats = cfg["features"]
        train = train_all.copy()
        if cfg["drop_missing_train"]:
            train = train.dropna(subset=[SOLIDITY])
        for target in TARGETS:
            m, model, _, yt, yp = fit_eval(
                train, test_known, feats, target, impute=cfg["impute"])
            m.update(variant=vname, target=TARGETS[target])
            rows.append(m)
            stash[(vname, target)] = (yt, yp, model, feats)

    comp = pd.DataFrame(rows)[["variant", "target", "R2", "RMSE", "MAE",
                               "n_train", "n_test"]]
    comp.to_csv(os.path.join(TABLES, "model_comparison.csv"), index=False)
    print("\n================ MODEL COMPARISON (fair test subset) ============")
    for tgt in TARGETS.values():
        print(f"\n  Target = {tgt}")
        sub = comp[comp.target == tgt]
        print(sub[["variant", "R2", "RMSE", "MAE", "n_train"]]
              .to_string(index=False))

    # ---- Coverage note ---------------------------------------------------
    # The missing solidity sits in the TRAINING data (many 2-blade props have
    # no geometry). Variant A discards those rows and trains on far fewer.
    n_train_full = len(train_all)
    n_train_A = len(train_all.dropna(subset=[SOLIDITY]))
    print("\nTraining-data cost of each strategy:")
    print(f"  A (complete-case) trains on {n_train_A} of {n_train_full} 2-blade rows "
          f"({100*n_train_A/n_train_full:.0f}%) - discards {n_train_full-n_train_A} rows "
          f"and, in deployment, cannot score any propeller lacking geometry.")
    print(f"  B (imputation)    trains on {n_train_full} of {n_train_full} rows (100%).")
    print(f"  C (no solidity)   trains on {n_train_full} of {n_train_full} rows (100%); "
          f"needs no geometry at all, so it can score every propeller.")

    # ---- Figure 6: R2 comparison bar chart -------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot = comp.pivot(index="target", columns="variant", values="R2")
    pivot = pivot.reindex(["C_T", "C_P", "eta"])
    pivot.plot(kind="bar", ax=ax, width=0.75,
               color=["#4C72B0", "#55A868", "#DD8452"])
    ax.set_ylabel("R^2 on held-out 3 & 4-blade propellers")
    ax.set_xlabel("Target")
    ax.set_title("Gradient Boosting - three missing-value strategies compared",
                 fontweight="bold")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(axis="x", rotation=0)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "06_model_comparison_r2.png"))
    plt.close(fig)

    # ---- Figure 7: predicted vs actual for variant B ---------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    vB = "B. With imputation (median solidity)"
    for a, (target, short) in zip(ax, TARGETS.items()):
        yt, yp, _, _ = stash[(vB, target)]
        a.scatter(yt, yp, s=6, alpha=0.3, color="#4C72B0")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        a.plot([lo, hi], [lo, hi], "r--", lw=1)
        r2 = r2_score(yt, yp)
        a.set_xlabel(f"actual {short}")
        a.set_ylabel(f"predicted {short}")
        a.set_title(f"{short}   (R^2 = {r2:.3f})")
    fig.suptitle("Predicted vs actual - variant B (with imputation), "
                 "held-out 3 & 4-blade props", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "07_pred_vs_actual.png"))
    plt.close(fig)

    # ---- Figure 8: feature importance (variant B, all three targets) -----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (target, short) in zip(ax, TARGETS.items()):
        _, _, model, feats = stash[(vB, target)]
        imp = pd.Series(model.feature_importances_, index=feats).sort_values()
        a.barh(imp.index, imp.values, color="#4C72B0")
        a.set_title(f"Feature importance - {short}")
        a.tick_params(axis="y", labelsize=8)
    fig.suptitle("Gradient Boosting feature importances (variant B)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "08_feature_importance.png"))
    plt.close(fig)

    print("\nSaved model_comparison.csv and figures 06-08.")


if __name__ == "__main__":
    main()
