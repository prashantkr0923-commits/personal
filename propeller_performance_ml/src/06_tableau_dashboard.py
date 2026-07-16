"""
06_tableau_dashboard.py  -  Week 2 (Tableau / data storytelling)
================================================================
Two deliverables for the Tableau task:

1. A clean Tableau EXTRACT  (data/processed/tableau_extract.csv) enriched with
   the fields a business dashboard needs: pitch-to-diameter ratio, working-state
   flag, clipped efficiency, per-propeller peak efficiency and the advance ratio
   at which it occurs, blade class, and solidity.

2. A rendered, storytelling DASHBOARD image (outputs/figures/10_dashboard.png)
   that mirrors the Tableau dashboard described in report/tableau_dashboard_spec.md
   - a KPI banner plus five coordinated views that walk a reader from "what do we
   have" to "what drives efficiency" to "the design trade-off".
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import PROCESSED, FIGURES

BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"


def build_extract():
    df = pd.read_csv(os.path.join(PROCESSED, "experiment_with_solidity.csv"))
    df["pitch_to_diameter"] = df["propeller_pitch"] / df["propeller_diameter"]
    df["working_state"] = np.where(df["thrust_coefficient_output"] > 0,
                                   "Working (thrust>0)", "Windmill/brake")
    df["efficiency_clipped"] = df["efficiency_output"].clip(-1.0, 0.9)
    df["blade_class"] = df["number_of_blades"].astype(int).astype(str) + "-blade"

    # per-propeller peak efficiency in the working state + where it occurs
    work = df[df["thrust_coefficient_output"] > 0]
    peak = (work.sort_values("efficiency_output")
                .groupby("propeller_name")
                .tail(1)[["propeller_name", "efficiency_output",
                          "advanced_ratio_input"]]
                .rename(columns={"efficiency_output": "peak_efficiency",
                                 "advanced_ratio_input": "j_at_peak_eff"}))
    df = df.merge(peak, on="propeller_name", how="left")

    out = os.path.join(PROCESSED, "tableau_extract.csv")
    df.to_csv(out, index=False)
    print(f"Saved Tableau extract -> {out}  {df.shape}")
    return df


def render_dashboard(df):
    prop = (df.groupby("propeller_name")
              .agg(brand=("propeller_brand", "first"),
                   blades=("number_of_blades", "first"),
                   diameter=("propeller_diameter", "first"),
                   pitch=("propeller_pitch", "first"),
                   pd_ratio=("pitch_to_diameter", "first"),
                   solidity=("solidity", "first"),
                   peak_eff=("peak_efficiency", "first"))
              .reset_index())

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.55, 1, 1],
                           hspace=0.42, wspace=0.28,
                           left=0.06, right=0.97, top=0.9, bottom=0.06)

    fig.suptitle("UAV Propeller Performance - Executive Dashboard",
                 fontsize=19, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925,
             "Which propellers deliver the most efficient thrust, and what "
             "geometry drives it?  |  240 propellers, 27,495 operating points",
             ha="center", fontsize=11, color="#555")

    # ---- KPI banner ------------------------------------------------------
    kpis = [
        ("Propellers tested", f"{prop['propeller_name'].nunique()}", BLUE),
        ("Operating points", f"{len(df):,}", BLUE),
        ("Best peak efficiency", f"{prop['peak_eff'].max():.2f}", GREEN),
        ("Median peak efficiency", f"{prop['peak_eff'].median():.2f}", ORANGE),
        ("Median solidity", f"{prop['solidity'].median():.3f}", BLUE),
    ]
    kpi_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :],
                                              wspace=0.15)
    for i, (label, value, color) in enumerate(kpis):
        a = fig.add_subplot(kpi_gs[0, i])
        a.axis("off")
        a.add_patch(plt.Rectangle((0, 0), 1, 1, transform=a.transAxes,
                                  facecolor=color, alpha=0.12))
        a.text(0.5, 0.62, value, ha="center", va="center", fontsize=22,
               fontweight="bold", color=color)
        a.text(0.5, 0.22, label, ha="center", va="center", fontsize=10,
               color="#444")

    # ---- View 1: efficiency envelope vs advance ratio --------------------
    a1 = fig.add_subplot(gs[1, 0])
    for _, sub in df.groupby("propeller_name"):
        sub = sub.sort_values("advanced_ratio_input")
        a1.plot(sub["advanced_ratio_input"], sub["efficiency_clipped"],
                color=BLUE, lw=0.4, alpha=0.15)
    a1.set_ylim(0, 0.9)
    a1.set_xlim(0, 1.6)
    a1.set_xlabel("Advance ratio  J")
    a1.set_ylabel("Efficiency")
    a1.set_title("1) Efficiency rises, peaks, then falls with advance ratio",
                 fontsize=11, fontweight="bold")

    # ---- View 2: peak efficiency by brand (top 12) -----------------------
    a2 = fig.add_subplot(gs[1, 1])
    brand_eff = (prop.groupby("brand")["peak_eff"].mean()
                     .sort_values(ascending=False).head(12))
    a2.barh(brand_eff.index[::-1], brand_eff.values[::-1], color=BLUE)
    a2.set_xlabel("Mean peak efficiency")
    a2.set_title("2) Peak efficiency by brand (top 12)",
                 fontsize=11, fontweight="bold")
    a2.tick_params(axis="y", labelsize=8)

    # ---- View 3: pitch/diameter vs peak efficiency -----------------------
    a3 = fig.add_subplot(gs[1, 2])
    cmap = {2: BLUE, 3: ORANGE, 4: GREEN}
    for b in sorted(prop["blades"].unique()):
        s = prop[prop["blades"] == b]
        a3.scatter(s["pd_ratio"], s["peak_eff"], s=22, alpha=0.7,
                   color=cmap[b], label=f"{b}-blade")
    a3.set_xlabel("Pitch / Diameter")
    a3.set_ylabel("Peak efficiency")
    a3.set_title("3) Higher pitch/diameter -> higher peak efficiency",
                 fontsize=11, fontweight="bold")
    a3.legend(fontsize=8)

    # ---- View 4: solidity vs peak efficiency -----------------------------
    a4 = fig.add_subplot(gs[2, 0])
    sol = prop.dropna(subset=["solidity"])
    for b in sorted(sol["blades"].unique()):
        s = sol[sol["blades"] == b]
        a4.scatter(s["solidity"], s["peak_eff"], s=22, alpha=0.7,
                   color=cmap[b], label=f"{b}-blade")
    a4.set_xlabel("Solidity")
    a4.set_ylabel("Peak efficiency")
    a4.set_title("4) Solidity vs efficiency (134 props with geometry)",
                 fontsize=11, fontweight="bold")
    a4.legend(fontsize=8)

    # ---- View 5: thrust vs power trade-off (density) ---------------------
    a5 = fig.add_subplot(gs[2, 1])
    work = df[df["thrust_coefficient_output"] > 0]
    hb = a5.hexbin(work["power_coefficient_output"],
                   work["thrust_coefficient_output"], gridsize=35,
                   cmap="Blues", mincnt=1)
    a5.set_xlabel("Power coefficient  C_P")
    a5.set_ylabel("Thrust coefficient  C_T")
    a5.set_title("5) Thrust-power trade-off (working state)",
                 fontsize=11, fontweight="bold")
    fig.colorbar(hb, ax=a5, shrink=0.8, label="rows")

    # ---- View 6: efficiency by blade class -------------------------------
    a6 = fig.add_subplot(gs[2, 2])
    order = sorted(prop["blades"].unique())
    data = [prop.loc[prop["blades"] == b, "peak_eff"].dropna() for b in order]
    bp = a6.boxplot(data, tick_labels=[f"{b}-blade" for b in order],
                    patch_artist=True)
    for patch, b in zip(bp["boxes"], order):
        patch.set_facecolor(cmap[b])
        patch.set_alpha(0.7)
    a6.set_ylabel("Peak efficiency")
    a6.set_title("6) Peak efficiency by blade count",
                 fontsize=11, fontweight="bold")

    fig.savefig(os.path.join(FIGURES, "10_dashboard.png"), dpi=110,
                facecolor="white")
    plt.close(fig)
    print("Saved dashboard -> outputs/figures/10_dashboard.png")


def main():
    df = build_extract()
    render_dashboard(df)


if __name__ == "__main__":
    main()
