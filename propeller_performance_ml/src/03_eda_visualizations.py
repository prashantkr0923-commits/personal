"""
03_eda_visualizations.py  -  Week 2 (Data Science), Part 3
==========================================================
Tasks addressed
---------------
1. Missing-value overview on the collated dataset.
2. Univariate analysis of the three performance outputs.
3. BIVARIATE analysis (the propeller performance curves and driver
   relationships) with written interpretation printed to the console.
4. Correlation HEATMAP between all numeric variables.

Outputs (outputs/figures/):
   02_target_distributions.png
   03_performance_curves.png
   04_bivariate_drivers.png
   05_correlation_heatmap.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED, FIGURES

sns.set_theme(style="whitegrid", font_scale=0.95)
PAL = {2: "#4C72B0", 3: "#DD8452", 4: "#55A868"}


def main():
    df = pd.read_csv(os.path.join(PROCESSED, "experiment_with_solidity.csv"))

    # ---------------------------------------------------------------- #
    # 1. Univariate distributions of the three targets
    # ---------------------------------------------------------------- #
    targets = ["thrust_coefficient_output", "power_coefficient_output",
               "efficiency_output"]
    titles = ["Thrust coefficient  C_T", "Power coefficient  C_P",
              "Efficiency  eta"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, t, title in zip(ax, targets, titles):
        sns.histplot(df[t], kde=True, ax=a, color="#4C72B0")
        a.set_title(title)
        a.set_xlabel("")
    fig.suptitle("Univariate distributions of performance outputs",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "02_target_distributions.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # 2. Bivariate: the classic propeller performance curves vs J
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, t, title in zip(ax, targets, titles):
        for name, sub in df.groupby("propeller_name"):
            sub = sub.sort_values("advanced_ratio_input")
            b = sub["number_of_blades"].iloc[0]
            a.plot(sub["advanced_ratio_input"], sub[t], lw=0.8, alpha=0.5,
                   color=PAL.get(b, "gray"))
        a.set_xlabel("Advance ratio  J")
        a.set_title(title + "  vs  J")
    # Efficiency explodes negative past zero-thrust (windmill state); clip the
    # view to the physically meaningful working band so the classic
    # rise-peak-fall efficiency curve is legible.
    ax[2].set_ylim(-0.2, 0.95)
    ax[0].set_ylabel("value")
    handles = [plt.Line2D([0], [0], color=PAL[b], label=f"{b} blades")
               for b in (2, 3, 4)]
    ax[2].legend(handles=handles, loc="upper right")
    fig.suptitle("Bivariate analysis - performance curves vs advance ratio",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "03_performance_curves.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # 3. Bivariate drivers: solidity / RPM / diameter vs performance
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    sns.scatterplot(data=df, x="solidity", y="thrust_coefficient_output",
                    hue="number_of_blades", palette=PAL, s=12, ax=ax[0],
                    legend="full")
    ax[0].set_title("C_T vs solidity")
    sns.scatterplot(data=df, x="solidity", y="power_coefficient_output",
                    hue="number_of_blades", palette=PAL, s=12, ax=ax[1],
                    legend=False)
    ax[1].set_title("C_P vs solidity")
    sns.boxplot(data=df, x="number_of_blades", y="efficiency_output",
                hue="number_of_blades", palette=PAL, ax=ax[2], legend=False)
    ax[2].set_title("Efficiency by blade count")
    fig.suptitle("Bivariate analysis - geometric & operational drivers",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "04_bivariate_drivers.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # 4. Correlation heatmap
    # ---------------------------------------------------------------- #
    num_cols = ["number_of_blades", "propeller_diameter", "propeller_pitch",
                "advanced_ratio_input", "rpm_rotation_input", "solidity",
                "total_blade_area", "disc_area",
                "thrust_coefficient_output", "power_coefficient_output",
                "efficiency_output"]
    corr = df[num_cols].corr()
    fig, a = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=a,
                annot_kws={"size": 7})
    a.set_title("Correlation heatmap of numeric variables", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "05_correlation_heatmap.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Printed interpretation (goes into the write-up)
    # ---------------------------------------------------------------- #
    print("Missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())
    print("\nCorrelation of drivers with efficiency:")
    print(corr["efficiency_output"].sort_values(ascending=False).round(2).to_string())
    print("\nCorrelation of drivers with thrust coefficient:")
    print(corr["thrust_coefficient_output"].sort_values(ascending=False).round(2).to_string())
    print("\nSaved 4 EDA figures to outputs/figures/")


if __name__ == "__main__":
    main()
