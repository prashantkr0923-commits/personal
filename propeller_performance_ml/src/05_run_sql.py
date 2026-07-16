"""
05_run_sql.py  -  Week 1 (SQL)
==============================
Loads the three raw experiment CSVs into an in-memory SQLite database
(experiment1 / experiment2 / experiment3) and answers the four Week-1 SQL
questions. The SQL itself lives in sql/week1_queries.sql; this runner executes
the key statements, prints the answers, and saves result tables/log.

Outputs:
   outputs/tables/sql_q2_efficiency_desc.csv
   outputs/tables/sql_q3_least100_power.csv
   outputs/tables/sql_answers.txt
   outputs/figures/09_sql_summary.png
"""
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RAW, TABLES, FIGURES

COLMAP = {
    "Propeller's Name": "propeller_name",
    "Blade's Name": "blade_name",
    "Propeller's Brand": "propeller_brand",
    "Number of Blades": "number_of_blades",
    "Propeller's Diameter": "propeller_diameter",
    "Propeller's Pitch": "propeller_pitch",
    "Advanced Ratio Input": "advance_ratio",
    "RPM Rotation Input": "rpm",
    "Thrust Coefficient Output": "thrust_coeff",
    "Power Coefficient Output": "power_coeff",
    "Efficiency Output": "efficiency",
}


def build_db():
    con = sqlite3.connect(":memory:")
    for vol in (1, 2, 3):
        df = pd.read_csv(os.path.join(RAW, f"Experiment_vol{vol}.csv"))
        df = df.rename(columns=COLMAP)
        df.to_sql(f"experiment{vol}", con, index=False, if_exists="replace")
    return con


def main():
    con = build_db()
    log = []

    def say(s):
        print(s)
        log.append(s)

    say("=" * 68)
    say("WEEK 1 - SQL ANSWERS")
    say("=" * 68)

    # Q1 -------------------------------------------------------------------
    q1 = pd.read_sql("SELECT COUNT(DISTINCT propeller_name) AS n "
                     "FROM experiment1 WHERE thrust_coeff > 0.12", con)["n"][0]
    q1_rows = pd.read_sql("SELECT COUNT(*) AS n FROM experiment1 "
                          "WHERE thrust_coeff > 0.12", con)["n"][0]
    say("\nQ1. Experiment 1 - propellers with thrust coefficient > 12% (0.12):")
    say(f"    distinct propellers : {q1}")
    say(f"    operating-point rows: {q1_rows}")

    # Q2 -------------------------------------------------------------------
    q2 = pd.read_sql(
        "SELECT propeller_name, advance_ratio, rpm, thrust_coeff, power_coeff, "
        "efficiency FROM experiment1 ORDER BY efficiency DESC", con)
    q2.to_csv(os.path.join(TABLES, "sql_q2_efficiency_desc.csv"), index=False)
    say("\nQ2. Experiment 1 reordered by efficiency (descending). "
        f"Saved {len(q2)} rows. Top 5:")
    say(q2.head(5).to_string(index=False))

    # Q3 -------------------------------------------------------------------
    q3 = pd.read_sql(
        "SELECT propeller_name, advance_ratio, rpm, power_coeff, efficiency "
        "FROM experiment1 ORDER BY power_coeff ASC LIMIT 100", con)
    q3.to_csv(os.path.join(TABLES, "sql_q3_least100_power.csv"), index=False)
    say("\nQ3. Experiment 1 - 100 least-performing rows by power coefficient. "
        "Saved 100 rows. Lowest 5:")
    say(q3.head(5).to_string(index=False))

    # Q4 -------------------------------------------------------------------
    merged_sql = ("WITH merged AS (SELECT * FROM experiment1 UNION ALL "
                  "SELECT * FROM experiment2 UNION ALL SELECT * FROM experiment3) ")
    q4 = pd.read_sql(merged_sql + "SELECT COUNT(DISTINCT propeller_name) AS n "
                     "FROM merged WHERE efficiency <= 0", con)["n"][0]
    q4_rows = pd.read_sql(merged_sql + "SELECT COUNT(*) AS n FROM merged "
                          "WHERE efficiency <= 0", con)["n"][0]
    total_props = pd.read_sql(merged_sql + "SELECT COUNT(DISTINCT propeller_name)"
                              " AS n FROM merged", con)["n"][0]
    say("\nQ4. Merged experiment 1+2+3 - propellers with efficiency <= 0:")
    say(f"    distinct propellers : {q4}  (of {total_props} total)")
    say(f"    operating-point rows: {q4_rows}")

    with open(os.path.join(TABLES, "sql_answers.txt"), "w") as f:
        f.write("\n".join(log))

    # ---- Figure 9: visual summary of the SQL answers ---------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].axis("off")
    text = (
        "WEEK 1  -  SQL ANSWERS\n"
        "--------------------------------------------\n\n"
        f"Q1  Experiment 1, thrust coeff > 12%\n"
        f"      distinct propellers : {q1}\n"
        f"      rows                : {q1_rows}\n\n"
        f"Q2  Experiment 1 sorted by efficiency DESC\n"
        f"      {len(q2)} rows  (see CSV)\n\n"
        f"Q3  100 least by power coefficient\n"
        f"      lowest CP = {q3['power_coeff'].min():.4f}\n\n"
        f"Q4  Merged 1+2+3, efficiency <= 0\n"
        f"      distinct propellers : {q4} of {total_props}\n"
        f"      rows                : {q4_rows}"
    )
    ax[0].text(0.02, 0.98, text, va="top", ha="left", family="monospace",
               fontsize=11)

    top10 = q2.head(10)
    ax[1].barh(range(len(top10))[::-1], top10["efficiency"], color="#4C72B0")
    ax[1].set_yticks(range(len(top10))[::-1])
    ax[1].set_yticklabels(top10["propeller_name"], fontsize=7)
    ax[1].set_xlabel("Efficiency")
    ax[1].set_title("Q2 - top 10 rows by efficiency (Experiment 1)")
    fig.suptitle("SQL task summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "09_sql_summary.png"))
    print("\nSaved SQL result tables and outputs/figures/09_sql_summary.png")


if __name__ == "__main__":
    main()
