"""
02_solidity_analysis.py  -  Week 2 (Data Science), Part 2
=========================================================
Tasks addressed
---------------
1. For each propeller, compute the AREA OF EACH BLADE and the TOTAL blade area.
   Blade area is the definite integral of the chord along the radius; it is
   evaluated with the composite trapezoidal rule via numpy.trapz.
2. Compute the DISC AREA of every propeller:  A = pi * r^2  (r = D/2).
3. Compute propeller SOLIDITY = (total blade area) / (disc area).
4. COLLATE the solidity values with the experiment data.
5. Check whether solidity could be found for every propeller and describe the
   findings, with a supporting visualization.

Outputs:
   data/processed/solidity_table.csv     - one row per propeller
   data/processed/experiment_with_solidity.csv
   outputs/tables/solidity_table.csv
   outputs/figures/01_solidity_findings.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED, FIGURES, TABLES

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def blade_area_trapz(sub):
    """Definite integral  A = integral of c(r) dr  over the blade span.

    Uses the composite trapezoidal rule (numpy.trapz) on the physical chord and
    radius distributions. Some blades in the raw data list each radial station
    more than once (repeated measurements); we average the chord per unique
    radius so the trapezoidal rule sees a clean, monotonically increasing span.
    Any NaN chords are linearly interpolated within the blade first, because the
    radial chord profile is smooth and integrating across a gap would bias the
    area low.
    """
    # collapse duplicate radial stations (mean chord per unique radius)
    sub = (sub.groupby("radius_distribution", as_index=False)["chord_distribution"]
              .mean()
              .sort_values("radius_distribution"))
    r = sub["radius_distribution"].to_numpy(dtype=float)
    c = sub["chord_distribution"].to_numpy(dtype=float)
    if len(r) < 2:
        return np.nan
    if np.isnan(c).any():
        good = ~np.isnan(c)
        if good.sum() < 2:
            return np.nan
        c = np.interp(r, r[good], c[good])   # fill interior gaps smoothly
    # numpy.trapz integrates y=c along x=r using the trapezoidal rule.
    return float(np.trapezoid(c, r)) if hasattr(np, "trapezoid") else float(np.trapz(c, r))


def main():
    geom = pd.read_csv(os.path.join(PROCESSED, "geometry_all.csv"))
    exp = pd.read_csv(os.path.join(PROCESSED, "experiment_all.csv"))

    # ---- 1. Area of each (single) blade, per blade name --------------------
    areas = (geom.groupby("blade_name")
                 .apply(blade_area_trapz, include_groups=False)
                 .rename("single_blade_area")
                 .reset_index())
    print("Single-blade areas computed for", len(areas), "blades")

    # ---- Propeller-level table: bring in blade count, diameter -------------
    prop_meta = (exp.groupby("propeller_name")
                    .agg(blade_name=("blade_name", "first"),
                         propeller_brand=("propeller_brand", "first"),
                         number_of_blades=("number_of_blades", "first"),
                         propeller_diameter=("propeller_diameter", "first"),
                         propeller_pitch=("propeller_pitch", "first"))
                    .reset_index())

    tab = prop_meta.merge(areas, on="blade_name", how="left")

    # ---- 1b. Total blade area = (#blades) x (single-blade area) ------------
    tab["total_blade_area"] = tab["number_of_blades"] * tab["single_blade_area"]

    # ---- 2. Disc area  A = pi * r^2 ---------------------------------------
    tab["disc_radius"] = tab["propeller_diameter"] / 2.0
    tab["disc_area"] = np.pi * tab["disc_radius"] ** 2

    # ---- 3. Solidity = total blade area / disc area -----------------------
    tab["solidity"] = tab["total_blade_area"] / tab["disc_area"]

    tab = tab.sort_values(["number_of_blades", "solidity"], na_position="last")
    tab.to_csv(os.path.join(PROCESSED, "solidity_table.csv"), index=False)
    tab.to_csv(os.path.join(TABLES, "solidity_table.csv"), index=False)

    # ---- 5. Findings: which propellers are missing solidity? --------------
    n_total = tab["propeller_name"].nunique()
    n_missing = int(tab["solidity"].isna().sum())
    missing_names = tab.loc[tab["solidity"].isna(), "propeller_name"].tolist()
    print("\n================ SOLIDITY FINDINGS ================")
    print(f"Propellers in experiment data : {n_total}")
    print(f"Solidity successfully computed : {n_total - n_missing}")
    print(f"Solidity MISSING (no geometry) : {n_missing}  -> {missing_names}")
    print(f"Solidity range  : {tab['solidity'].min():.4f} .. {tab['solidity'].max():.4f}")
    print(f"Solidity mean   : {tab['solidity'].mean():.4f}")
    print("By blade count (mean solidity):")
    print(tab.groupby('number_of_blades')['solidity'].mean().to_string())

    # ---- 4. Collate solidity onto the experiment data ---------------------
    exp_sol = exp.merge(tab[["propeller_name", "single_blade_area",
                             "total_blade_area", "disc_area", "solidity"]],
                        on="propeller_name", how="left")
    exp_sol.to_csv(os.path.join(PROCESSED, "experiment_with_solidity.csv"),
                   index=False)
    print(f"\nCollated experiment+solidity -> {exp_sol.shape}, "
          f"rows lacking solidity = {int(exp_sol['solidity'].isna().sum())}")

    # ---- Visualization: solidity findings ---------------------------------
    plotted = tab.dropna(subset=["solidity"])
    blade_levels = sorted(plotted["number_of_blades"].unique())

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # (a) coverage: how many propellers actually got a solidity value
    cov_labels = ["Solidity\ncomputed", "Solidity MISSING\n(no geometry)"]
    cov_vals = [n_total - n_missing, n_missing]
    ax[0].bar(cov_labels, cov_vals, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(cov_vals):
        ax[0].text(i, v + 2, str(v), ha="center", fontweight="bold")
    ax[0].set_ylabel("Number of propellers")
    ax[0].set_title(f"Solidity coverage\n{n_total-n_missing}/{n_total} propellers "
                    f"({100*(n_total-n_missing)/n_total:.0f}%)")

    # (b) distribution of computed solidity
    ax[1].hist(plotted["solidity"], bins=25, color="#4C72B0", edgecolor="white")
    ax[1].axvline(plotted["solidity"].mean(), color="red", ls="--",
                  label=f"mean={plotted['solidity'].mean():.3f}")
    ax[1].set_xlabel("Solidity  (total blade area / disc area)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Distribution of computed solidity")
    ax[1].legend()

    # (c) solidity vs blade count
    ax[2].boxplot([plotted.loc[plotted.number_of_blades == b, "solidity"]
                   for b in blade_levels],
                  tick_labels=[f"{b} blades" for b in blade_levels])
    ax[2].set_ylabel("Solidity")
    ax[2].set_title("Solidity by blade count")

    fig.suptitle(f"Solidity findings  ({n_missing} of {n_total} propellers had "
                 f"NO geometry → solidity N/A)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "01_solidity_findings.png"))
    print("\nSaved figure outputs/figures/01_solidity_findings.png")


if __name__ == "__main__":
    main()
