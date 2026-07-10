"""
Crime Analysis — Exploratory Data Analysis & Cluster Analysis
=============================================================
Course-End Project: Crime Analysis.

The original problem statement is framed as a Tableau dashboard exercise across
four dashboards plus an analytical (clustering) component. This script reproduces
that analysis programmatically in Python and produces downloadable outputs
(charts as PNG, summary tables as CSV/Excel, and a consolidated HTML report).

Tasks
-----
1. Overall Crime Statistics   — counts & types of crimes, geo distribution,
                                 most common incidents, live-feed snapshot.
2. Time-Period Analysis       — day-of-week / hour distribution, time-block %.
3. Trend Analysis             — crime rate change across years and months.
4. Comparative Analysis       — arrest vs. no-arrest, category breakdowns.
5. Cluster Analysis           — unsupervised segmentation of incidents and
                                 city crime-profile clustering.

Note on data: the CSV shipped with this project uses a simplified schema
(Crime ID, Date, Time, City, Location, Latitude, Longitude, Crime Type,
Number of Crimes, Number of Arrests, Arrest Made, Severity). It does not carry
the `Domestic` field referenced in the generic variable dictionary, so Task 4's
"domestic" breakdown is adapted to a Severity-category breakdown and this is
stated explicitly in the report.
"""

import os
import textwrap
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration & styling
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get(
    "CRIME_CSV",
    "/root/.claude/uploads/b3aea3b9-2b91-57c0-a4e2-6e2a84a55b0d/e8e008a4-Crime_Analysis_Dataset.csv",
)
OUT = os.path.join(HERE, "outputs")
CHARTS = os.path.join(OUT, "charts")
TABLES = os.path.join(OUT, "tables")
for d in (OUT, CHARTS, TABLES):
    os.makedirs(d, exist_ok=True)

# Colour-blind-safe categorical palette (Okabe–Ito) + sequential accents.
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
       "#56B4E9", "#F0E442", "#999999"]
INK = "#1a1a1a"
GRID = "#e6e6e6"
SEV_COLORS = {"Low": "#009E73", "Medium": "#E69F00", "High": "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style_ax(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    if title:
        ax.set_title(title, loc="left")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.7)
    ax.set_axisbelow(True)
    return ax


def save(fig, name):
    path = os.path.join(CHARTS, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  chart  -> outputs/charts/{name}")
    return path


def dump_table(df, name, index=True):
    path = os.path.join(TABLES, name)
    df.to_csv(path, index=index)
    print(f"  table  -> outputs/tables/{name}")
    return path


# ----------------------------------------------------------------------------
# Load & feature engineering
# ----------------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Parse HH:MM time
    t = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    df["Hour"] = t.dt.hour
    df["Minute"] = t.dt.minute
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["DOW"] = df["Date"].dt.day_name()
    df["DOW_num"] = df["Date"].dt.dayofweek
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["IsWeekend"] = df["DOW_num"] >= 5

    def block(h):
        if 5 <= h < 12:
            return "Morning"
        if 12 <= h < 17:
            return "Afternoon"
        if 17 <= h < 21:
            return "Evening"
        return "Night"
    df["TimeBlock"] = df["Hour"].apply(block)
    df["ArrestFlag"] = (df["Arrest Made"] == "Arrest Made").astype(int)
    df["SeverityOrd"] = df["Severity"].map({"Low": 1, "Medium": 2, "High": 3})
    return df


DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
BLOCK_ORDER = ["Morning", "Afternoon", "Evening", "Night"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

REPORT_SECTIONS = []  # (title, intro, [(img_path, caption)], [insight,...])


# ============================================================================
# TASK 1 — Overall Crime Statistics
# ============================================================================
def task1_overall(df):
    print("\n[Task 1] Overall Crime Statistics")
    imgs, insights = [], []

    # Crime type counts
    ct = df["Crime Type"].value_counts()
    dump_table(ct.rename("Incidents").to_frame(), "t1_crime_type_counts.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ct.index[::-1], ct.values[::-1], color=PAL[0])
    for i, v in enumerate(ct.values[::-1]):
        ax.text(v + 1, i, str(v), va="center", fontsize=9)
    style_ax(ax, "Most common criminal incidents", "Number of incidents", "",
             grid_axis="x")
    imgs.append((save(fig, "t1_crime_types.png"),
                 "Incident volume by crime type across all cities."))

    # City counts
    cc = df["City"].value_counts()
    dump_table(cc.rename("Incidents").to_frame(), "t1_city_counts.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(cc.index, cc.values, color=PAL[5])
    ax.set_xticklabels(cc.index, rotation=30, ha="right")
    for i, v in enumerate(cc.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    style_ax(ax, "Reported incidents by city", "", "Number of incidents")
    imgs.append((save(fig, "t1_city_counts.png"),
                 "Total reported incidents by city."))

    # Severity split
    sev = df["Severity"].value_counts().reindex(["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(sev.index, sev.values, color=[SEV_COLORS[s] for s in sev.index])
    for i, v in enumerate(sev.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=10)
    style_ax(ax, "Incident severity distribution", "", "Number of incidents")
    imgs.append((save(fig, "t1_severity.png"),
                 "How incidents split across severity bands."))

    # Geo map (city-level bubbles sized by volume, coloured by high-severity share)
    geo = (df.groupby("City")
             .agg(Incidents=("Crime ID", "count"),
                  Lat=("Latitude", "mean"), Lon=("Longitude", "mean"),
                  HighSevShare=("Severity", lambda s: (s == "High").mean()),
                  ArrestRate=("ArrestFlag", "mean"))
             .reset_index())
    dump_table(geo, "t1_geo_city_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(geo["Lon"], geo["Lat"], s=geo["Incidents"] * 6,
                    c=geo["HighSevShare"], cmap="YlOrRd",
                    edgecolor=INK, linewidth=0.8, alpha=0.9,
                    vmin=0.15, vmax=0.30)
    for _, r in geo.iterrows():
        ax.annotate(f"{r['City']}\n({int(r['Incidents'])})",
                    (r["Lon"], r["Lat"]), fontsize=8, ha="center",
                    va="center", xytext=(0, 0), textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of high-severity incidents")
    style_ax(ax, "Geographic distribution of crime (bubble = volume)",
             "Longitude", "Latitude", grid_axis="both")
    ax.grid(alpha=0.3)
    imgs.append((save(fig, "t1_geo_map.png"),
                 "City locations sized by incident volume and coloured by the "
                 "share of high-severity incidents."))

    # Live-feed snapshot (text KPIs)
    cur_year = int(df["Year"].max())
    ytd = int((df["Year"] == cur_year).sum())
    recent = df.sort_values(["Date", "Hour", "Minute"]).iloc[-1]
    top_crime = ct.index[0]
    kpi = pd.DataFrame({
        "Metric": ["Total incidents (all years)",
                   f"Incidents reported in {cur_year}",
                   "Most common crime type",
                   "Cities covered",
                   "Most recent incident"],
        "Value": [len(df), ytd, f"{top_crime} ({ct.iloc[0]})",
                  df["City"].nunique(),
                  f"{recent['Crime Type']} @ {recent['City']} "
                  f"({recent['Location']}) on "
                  f"{recent['Date'].date()} {recent['Time']}"],
    })
    dump_table(kpi, "t1_live_feed_kpis.csv", index=False)

    insights = [
        f"{len(df):,} incidents span {df['City'].nunique()} cities and "
        f"{df['Crime Type'].nunique()} crime types over "
        f"{df['Date'].min().date()} to {df['Date'].max().date()}.",
        f"The most common incident is <b>{top_crime}</b> ({ct.iloc[0]} cases), "
        f"followed by {ct.index[1]} and {ct.index[2]}.",
        f"<b>{sev['High']}</b> incidents ({sev['High']/len(df):.0%}) are "
        f"high-severity; the volume leader is <b>{cc.index[0]}</b> "
        f"({cc.iloc[0]} incidents).",
        f"Live-feed snapshot: <b>{ytd}</b> incidents logged in {cur_year}; most "
        f"recent = {recent['Crime Type']} in {recent['City']} on "
        f"{recent['Date'].date()} at {recent['Time']}.",
    ]
    REPORT_SECTIONS.append((
        "Task 1 — Overall Crime Statistics",
        "Volume, mix and geography of reported incidents, plus a live-feed "
        "snapshot of the most recent activity.",
        imgs, insights))
    return geo


# ============================================================================
# TASK 2 — Time-Period Analysis
# ============================================================================
def task2_time(df):
    print("\n[Task 2] Time-Period Analysis")
    imgs, insights = [], []

    dow = df["DOW"].value_counts().reindex(DOW_ORDER)
    dump_table(dow.rename("Incidents").to_frame(), "t2_day_of_week.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PAL[3] if d in ("Saturday", "Sunday") else PAL[0] for d in DOW_ORDER]
    ax.bar(range(7), dow.values, color=colors)
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in DOW_ORDER])
    for i, v in enumerate(dow.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    style_ax(ax, "Incidents by day of week (weekend highlighted)", "",
             "Number of incidents")
    imgs.append((save(fig, "t2_day_of_week.png"),
                 "Weekday vs. weekend incident volume."))

    hr = df["Hour"].value_counts().reindex(range(24), fill_value=0)
    dump_table(hr.rename("Incidents").to_frame(), "t2_hour_of_day.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(hr.index, hr.values, color=PAL[0], alpha=0.18)
    ax.plot(hr.index, hr.values, color=PAL[0], lw=2, marker="o", ms=4)
    ax.set_xticks(range(0, 24, 2))
    style_ax(ax, "Incidents by hour of day", "Hour (24h)", "Number of incidents")
    imgs.append((save(fig, "t2_hour_of_day.png"),
                 "Hourly reporting curve across the full period."))

    blk = df["TimeBlock"].value_counts().reindex(BLOCK_ORDER)
    blk_pct = (blk / blk.sum() * 100).round(1)
    tb = pd.concat([blk.rename("Incidents"), blk_pct.rename("Percent")], axis=1)
    dump_table(tb, "t2_time_blocks.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    wed, _, aut = ax.pie(
        blk.values, labels=BLOCK_ORDER, autopct="%1.1f%%",
        colors=[PAL[1], PAL[2], PAL[4], PAL[6]],
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        pctdistance=0.79, startangle=90)
    ax.set_title("Share of incidents by time block", loc="center",
                 fontweight="bold")
    ax.text(0, 0, f"{len(df):,}\nincidents", ha="center", va="center",
            fontsize=12, fontweight="bold")
    imgs.append((save(fig, "t2_time_blocks.png"),
                 "Morning 05–12, Afternoon 12–17, Evening 17–21, Night 21–05."))

    # Heatmap DOW x Hour
    piv = (df.pivot_table(index="DOW", columns="Hour", values="Crime ID",
                          aggfunc="count", fill_value=0)
             .reindex(DOW_ORDER).reindex(columns=range(24), fill_value=0))
    dump_table(piv, "t2_heatmap_dow_hour.csv")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(7))
    ax.set_yticklabels([d[:3] for d in DOW_ORDER])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Incidents")
    style_ax(ax, "When do incidents happen? (day × hour)", "Hour of day", "")
    ax.grid(False)
    imgs.append((save(fig, "t2_heatmap.png"),
                 "Density of incidents by weekday and hour."))

    peak_block = blk_pct.idxmax()
    peak_hour = int(hr.idxmax())
    peak_dow = dow.idxmax()
    insights = [
        f"Busiest day is <b>{peak_dow}</b> ({int(dow.max())} incidents); "
        f"weekend share is "
        f"{df['IsWeekend'].mean():.0%} of all incidents.",
        f"Reporting peaks around <b>{peak_hour:02d}:00</b> "
        f"({int(hr.max())} incidents in that hour).",
        f"<b>{peak_block}</b> is the dominant time block at "
        f"{blk_pct.max():.1f}% of incidents; the split is "
        + ", ".join(f"{b} {blk_pct[b]:.0f}%" for b in BLOCK_ORDER) + ".",
    ]
    REPORT_SECTIONS.append((
        "Task 2 — Time-Period Analysis",
        "How incidents distribute across day of week, hour, and named time "
        "blocks — the timing signal behind preventive deployment.",
        imgs, insights))


# ============================================================================
# TASK 3 — Trend Analysis
# ============================================================================
def task3_trend(df):
    print("\n[Task 3] Trend Analysis")
    imgs, insights = [], []

    ym = df.groupby("YearMonth").size()
    dump_table(ym.rename("Incidents").to_frame(), "t3_monthly_trend.csv")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(range(len(ym)), ym.values, color=PAL[0], lw=2, marker="o", ms=4)
    step = max(1, len(ym) // 12)
    ax.set_xticks(range(0, len(ym), step))
    ax.set_xticklabels(ym.index[::step], rotation=45, ha="right")
    style_ax(ax, "Monthly incident trend", "", "Number of incidents")
    imgs.append((save(fig, "t3_monthly_trend.png"),
                 "Reported incidents per calendar month."))

    # Year comparison by month
    yr_month = (df.pivot_table(index="Month", columns="Year", values="Crime ID",
                               aggfunc="count", fill_value=0)
                  .reindex(range(1, 13), fill_value=0))
    yr_month.index = MONTH_ORDER
    dump_table(yr_month, "t3_year_month_matrix.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, y in enumerate(yr_month.columns):
        ax.plot(MONTH_ORDER, yr_month[y].values, marker="o", ms=4,
                lw=2, color=PAL[i % len(PAL)], label=str(y))
    ax.legend(title="Year", frameon=False)
    style_ax(ax, "Same-month comparison across years", "", "Number of incidents")
    imgs.append((save(fig, "t3_year_month_compare.png"),
                 "Month-by-month incident counts overlaid by year (partial "
                 "first/last years reflect the data window)."))

    yr = df["Year"].value_counts().sort_index()
    dump_table(yr.rename("Incidents").to_frame(), "t3_yearly_totals.csv")

    # Crime-type trend by year (share)
    ty = (df.pivot_table(index="Year", columns="Crime Type", values="Crime ID",
                         aggfunc="count", fill_value=0))
    ty_share = ty.div(ty.sum(axis=1), axis=0) * 100
    dump_table(ty_share.round(1), "t3_crimetype_share_by_year.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(len(ty_share))
    for i, c in enumerate(ty_share.columns):
        ax.bar(ty_share.index.astype(str), ty_share[c].values, bottom=bottom,
               color=PAL[i % len(PAL)], label=c, width=0.6)
        bottom += ty_share[c].values
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False,
              fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter())
    style_ax(ax, "Crime-type mix by year (share of incidents)", "",
             "Share of incidents")
    imgs.append((save(fig, "t3_crimetype_mix.png"),
                 "Composition of incidents by crime type within each year."))

    yrs = sorted(df["Year"].dropna().unique())
    trend_note = ", ".join(f"{int(y)}: {int(yr.get(y, 0))}" for y in yrs)
    insights = [
        f"Yearly incident counts — {trend_note}. Note the first and last "
        f"calendar years are partial given the "
        f"{df['Date'].min().date()}–{df['Date'].max().date()} window.",
        f"Monthly volume averages {ym.mean():.0f} incidents "
        f"(min {ym.min()}, max {ym.max()}), with no runaway seasonal spike — "
        f"consistent with an evenly sampled dataset.",
        "Crime-type mix stays broadly stable across years, so shifts are in "
        "volume/timing rather than the type of crime being committed.",
    ]
    REPORT_SECTIONS.append((
        "Task 3 — Trend Analysis",
        "Change in reporting over time: monthly trajectory, same-month "
        "year-over-year comparison, and how the crime-type mix evolves.",
        imgs, insights))


# ============================================================================
# TASK 4 — Comparative Analysis
# ============================================================================
def task4_compare(df):
    print("\n[Task 4] Comparative Analysis")
    imgs, insights = [], []

    # Arrest vs no arrest
    arr = df["Arrest Made"].value_counts()
    dump_table(arr.rename("Incidents").to_frame(), "t4_arrest_split.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.pie(arr.values, labels=arr.index, autopct="%1.1f%%",
           colors=[PAL[2], PAL[3]], startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("Arrest outcome distribution", loc="center", fontweight="bold")
    imgs.append((save(fig, "t4_arrest_split.png"),
                 "Overall share of incidents resulting in an arrest."))

    # Arrest rate by crime type
    rate = (df.groupby("Crime Type")["ArrestFlag"].mean()
              .sort_values(ascending=False) * 100)
    dump_table(rate.round(1).rename("ArrestRate%").to_frame(),
               "t4_arrest_rate_by_type.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(rate.index[::-1], rate.values[::-1], color=PAL[0])
    for i, v in enumerate(rate.values[::-1]):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter())
    style_ax(ax, "Arrest rate by crime type", "Arrest rate", "", grid_axis="x")
    imgs.append((save(fig, "t4_arrest_rate_by_type.png"),
                 "Percentage of incidents per crime type ending in arrest."))

    # Severity mix by crime type (adapted 'category' breakdown; no Domestic field)
    sev_ct = (df.pivot_table(index="Crime Type", columns="Severity",
                             values="Crime ID", aggfunc="count", fill_value=0)
                .reindex(columns=["Low", "Medium", "High"], fill_value=0))
    sev_share = sev_ct.div(sev_ct.sum(axis=1), axis=0) * 100
    sev_share = sev_share.sort_values("High", ascending=False)
    dump_table(sev_share.round(1), "t4_severity_share_by_type.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(sev_share))
    for s in ["Low", "Medium", "High"]:
        ax.bar(sev_share.index, sev_share[s].values, bottom=bottom,
               color=SEV_COLORS[s], label=s, width=0.62)
        bottom += sev_share[s].values
    ax.set_xticklabels(sev_share.index, rotation=30, ha="right")
    ax.legend(title="Severity", frameon=False, bbox_to_anchor=(1.01, 1),
              loc="upper left")
    ax.yaxis.set_major_formatter(PercentFormatter())
    style_ax(ax, "Severity composition within each crime type", "",
             "Share of incidents")
    imgs.append((save(fig, "t4_severity_by_type.png"),
                 "Per-category severity mix — the closest available proxy for "
                 "the 'domestic share' breakdown, since the dataset carries no "
                 "Domestic flag."))

    # Arrest rate by city + severity
    city_arr = (df.groupby("City")["ArrestFlag"].mean() * 100).sort_values()
    dump_table(city_arr.round(1).rename("ArrestRate%").to_frame(),
               "t4_arrest_rate_by_city.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(city_arr.index, city_arr.values, color=PAL[5])
    for i, v in enumerate(city_arr.values):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter())
    style_ax(ax, "Arrest rate by city", "Arrest rate", "", grid_axis="x")
    imgs.append((save(fig, "t4_arrest_rate_by_city.png"),
                 "How arrest outcomes vary across cities."))

    top_rate = rate.index[0]
    low_rate = rate.index[-1]
    insights = [
        f"<b>{arr.get('Arrest Made', 0)}</b> incidents "
        f"({df['ArrestFlag'].mean():.0%}) resulted in an arrest vs. "
        f"{arr.get('No Arrest', 0)} with no arrest.",
        f"Arrest rate is highest for <b>{top_rate}</b> ({rate.iloc[0]:.0f}%) "
        f"and lowest for <b>{low_rate}</b> ({rate.iloc[-1]:.0f}%).",
        f"High-severity share peaks for <b>{sev_share.index[0]}</b> "
        f"({sev_share['High'].iloc[0]:.0f}% high). The dataset has no "
        f"<i>Domestic</i> field, so this severity breakdown stands in for the "
        f"requested domestic-category analysis.",
        f"City arrest rates range from {city_arr.min():.0f}% "
        f"({city_arr.index[0]}) to {city_arr.max():.0f}% "
        f"({city_arr.index[-1]}).",
    ]
    REPORT_SECTIONS.append((
        "Task 4 — Comparative Analysis",
        "Arrest outcomes compared across crime type and city, plus the "
        "severity composition of each category (adapted from the requested "
        "domestic breakdown, which the dataset cannot support).",
        imgs, insights))


# ============================================================================
# TASK 5 — Cluster Analysis
# ============================================================================
def task5_cluster(df, geo):
    print("\n[Task 5] Cluster Analysis")
    imgs, insights = [], []

    # ---- 5a. Incident-level K-means -------------------------------------
    # Cluster on the *circumstances* of an incident — when, what day, how
    # severe and where. Arrest outcome is deliberately excluded from the
    # inputs (it is near-collinear with Number of Arrests and would otherwise
    # trivially split the data into arrest/no-arrest); instead we profile the
    # arrest rate as an *outcome* of each discovered segment.
    feat = pd.DataFrame({
        "Hour": df["Hour"],
        "DOW": df["DOW_num"],
        "Severity": df["SeverityOrd"],
        "Lat": df["Latitude"],
        "Lon": df["Longitude"],
    }).dropna()
    X = StandardScaler().fit_transform(feat)

    ks = range(2, 11)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))
    diag = pd.DataFrame({"k": list(ks), "inertia": inertias,
                         "silhouette": np.round(sils, 4)})
    dump_table(diag, "t5_kmeans_diagnostics.csv", index=False)

    best_k = int(ks[int(np.argmax(sils))])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(list(ks), inertias, marker="o", color=PAL[0], lw=2)
    style_ax(a1, "Elbow (inertia vs. k)", "k", "Inertia")
    a2.plot(list(ks), sils, marker="o", color=PAL[3], lw=2)
    a2.axvline(best_k, color=PAL[2], ls="--", lw=1.5)
    a2.text(best_k, min(sils), f" best k={best_k}", color=PAL[2], fontsize=9)
    style_ax(a2, "Silhouette vs. k", "k", "Silhouette score")
    imgs.append((save(fig, "t5_kmeans_diagnostics.png"),
                 "Choosing the number of incident clusters. Silhouette is "
                 f"maximised at k={best_k}."))

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    df_c = df.loc[feat.index].copy()
    df_c["Cluster"] = km.labels_

    pca = PCA(n_components=2, random_state=42).fit(X)
    P = pca.transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(best_k):
        m = km.labels_ == c
        ax.scatter(P[m, 0], P[m, 1], s=22, alpha=0.7,
                   color=PAL[c % len(PAL)], label=f"Cluster {c}")
    cen = pca.transform(km.cluster_centers_)
    ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=180, c="black",
               edgecolor="white", linewidth=1.4, zorder=5)
    ax.legend(frameon=False, fontsize=9)
    var = pca.explained_variance_ratio_
    style_ax(ax, f"Incident clusters in PCA space (k={best_k})",
             f"PC1 ({var[0]:.0%} var)", f"PC2 ({var[1]:.0%} var)",
             grid_axis="both")
    ax.grid(alpha=0.3)
    imgs.append((save(fig, "t5_pca_clusters.png"),
                 "Incidents projected to two principal components and coloured "
                 "by cluster; black X = cluster centroids."))

    # Cluster profiles
    prof = (df_c.groupby("Cluster")
                 .agg(Size=("Crime ID", "count"),
                      AvgHour=("Hour", "mean"),
                      AvgSeverity=("SeverityOrd", "mean"),
                      ArrestRate=("ArrestFlag", "mean"),
                      AvgArrests=("Number of Arrests", "mean"),
                      TopCrime=("Crime Type",
                                lambda s: s.value_counts().idxmax()),
                      TopCity=("City", lambda s: s.value_counts().idxmax()),
                      WeekendShare=("IsWeekend", "mean"))
                 .round(2))
    prof["ArrestRate"] = (prof["ArrestRate"] * 100).round(0)
    prof["WeekendShare"] = (prof["WeekendShare"] * 100).round(0)
    dump_table(prof, "t5_cluster_profiles.csv")

    # Save incident-level assignments
    keep = ["Crime ID", "Date", "Time", "City", "Location", "Crime Type",
            "Severity", "Arrest Made", "Number of Arrests", "Hour", "DOW",
            "TimeBlock", "Cluster"]
    dump_table(df_c[keep], "t5_incident_cluster_assignments.csv", index=False)

    # Heatmap of cluster feature means (standardized)
    feat_c = feat.copy()
    feat_c["Cluster"] = km.labels_
    means = feat_c.groupby("Cluster").mean()
    means_z = (means - feat.mean()) / feat.std()
    fig, ax = plt.subplots(figsize=(8.5, 0.7 * best_k + 2))
    im = ax.imshow(means_z.values, cmap="RdBu_r", aspect="auto",
                   vmin=-2, vmax=2)
    ax.set_xticks(range(len(means_z.columns)))
    ax.set_xticklabels(means_z.columns, rotation=30, ha="right")
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"Cluster {c}" for c in range(best_k)])
    for i in range(best_k):
        for j in range(len(means_z.columns)):
            ax.text(j, i, f"{means_z.values[i, j]:+.1f}", ha="center",
                    va="center", fontsize=8,
                    color="white" if abs(means_z.values[i, j]) > 1.2 else INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="z-score vs. mean")
    style_ax(ax, "Cluster fingerprints (standardised feature means)", "", "")
    ax.grid(False)
    imgs.append((save(fig, "t5_cluster_fingerprints.png"),
                 "Each cluster's feature profile relative to the overall mean "
                 "(red = above average, blue = below)."))

    # ---- 5b. City crime-profile hierarchical clustering -----------------
    city_prof = (df.pivot_table(index="City", columns="Crime Type",
                                values="Crime ID", aggfunc="count",
                                fill_value=0))
    city_prof_share = city_prof.div(city_prof.sum(axis=1), axis=0)
    dump_table(city_prof_share.round(3), "t5_city_crime_profile.csv")
    Z = linkage(city_prof_share.values, method="ward")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dendrogram(Z, labels=city_prof_share.index.tolist(), ax=ax,
               color_threshold=0.6 * max(Z[:, 2]),
               above_threshold_color="#999999")
    style_ax(ax, "City clustering by crime-type profile (Ward linkage)", "",
             "Distance")
    ax.grid(False)
    imgs.append((save(fig, "t5_city_dendrogram.png"),
                 "Cities grouped by the similarity of their crime-type mix."))

    n_city_clusters = 3
    city_labels = AgglomerativeClustering(
        n_clusters=n_city_clusters, linkage="ward").fit_predict(
        city_prof_share.values)
    city_out = city_prof_share.copy()
    city_out.insert(0, "Cluster", city_labels)
    dump_table(city_out.round(3), "t5_city_cluster_assignments.csv")

    sil_best = max(sils)
    biggest = int(prof["Size"].idxmax())
    insights = [
        f"K-means on 5 circumstance features (hour, day, severity, lat, long) "
        f"favours <b>k={best_k}</b> (silhouette {sil_best:.2f}); the two "
        f"principal components capture {var[0]+var[1]:.0%} of variance, driven "
        f"largely by geography (lat/long) and time-of-day.",
        f"The largest segment is <b>Cluster {biggest}</b> "
        f"({int(prof.loc[biggest,'Size'])} incidents, top crime "
        f"{prof.loc[biggest,'TopCrime']} in {prof.loc[biggest,'TopCity']}, "
        f"clearance/arrest rate {prof.loc[biggest,'ArrestRate']:.0f}%).",
        "Arrest outcome was held out of the clustering and profiled afterwards: "
        f"clearance is essentially uniform (~{prof['ArrestRate'].mean():.0f}%) "
        "across the geographic segments, i.e. clearance is not "
        "location-driven in this data — a useful negative result for resource "
        "planning.",
        "Because location dominates the five features, the K-means segments are "
        "effectively regional; the silhouette also rises again at higher k "
        "(city-level granularity), so k is a resolution choice between broad "
        "regions and individual cities rather than a single 'true' number.",
        f"Hierarchical clustering of cities on their crime-type mix yields "
        f"{n_city_clusters} city groups, useful for peer benchmarking and "
        f"shared prevention strategy.",
    ]
    REPORT_SECTIONS.append((
        "Task 5 — Cluster Analysis",
        "Unsupervised segmentation: K-means groups individual incidents by "
        "when/where/how-severe they are, and hierarchical clustering groups "
        "cities by their crime-type signature.",
        imgs, insights))
    return prof, diag


# ============================================================================
# Excel workbook + HTML report
# ============================================================================
def build_excel(df):
    print("\n[Export] Excel workbook")
    path = os.path.join(OUT, "Crime_Analysis_Summary.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for fname in sorted(os.listdir(TABLES)):
            if not fname.endswith(".csv"):
                continue
            sheet = os.path.splitext(fname)[0][:31]
            has_index = not fname.endswith(
                ("assignments.csv", "counts.csv", "kpis.csv",
                 "summary.csv", "diagnostics.csv", "split.csv"))
            t = pd.read_csv(os.path.join(TABLES, fname))
            t.to_excel(xl, sheet_name=sheet, index=False)
    print(f"  xlsx   -> outputs/Crime_Analysis_Summary.xlsx")
    return path


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html(df):
    print("\n[Export] HTML report")
    css = """
    :root{--ink:#1a1a1a;--muted:#5a5a5a;--line:#e6e6e6;--accent:#0072B2;--bg:#f7f8fa;}
    *{box-sizing:border-box}
    body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);line-height:1.6;}
    .wrap{max-width:1080px;margin:0 auto;padding:0 24px 80px;}
    header{background:linear-gradient(135deg,#0b2545,#134074);color:#fff;padding:56px 24px;margin-bottom:8px;}
    header .wrap{padding-bottom:0}
    h1{margin:0 0 8px;font-size:34px;letter-spacing:-0.5px}
    header p{margin:4px 0;color:#cfe0f5;font-size:15px}
    .kpis{display:flex;flex-wrap:wrap;gap:14px;margin:28px 0 8px;}
    .kpi{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;}
    .kpi .n{font-size:26px;font-weight:700;color:var(--accent)}
    .kpi .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
    nav{position:sticky;top:0;background:rgba(247,248,250,.92);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:12px 0;z-index:10;margin-bottom:24px}
    nav a{color:var(--accent);text-decoration:none;font-size:13px;font-weight:600;margin-right:18px;white-space:nowrap}
    section{background:#fff;border:1px solid var(--line);border-radius:16px;padding:28px 30px;margin:26px 0;}
    section h2{margin:0 0 6px;font-size:23px}
    .intro{color:var(--muted);margin:0 0 18px;font-size:15px}
    .fig{margin:22px 0}
    .fig img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}
    .cap{font-size:13px;color:var(--muted);margin-top:8px}
    .insights{background:var(--bg);border-left:4px solid var(--accent);border-radius:8px;padding:14px 18px;margin-top:18px}
    .insights h3{margin:0 0 8px;font-size:14px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
    .insights ul{margin:0;padding-left:20px}
    .insights li{margin:6px 0;font-size:14.5px}
    footer{color:var(--muted);font-size:13px;text-align:center;padding:30px 0}
    .badge{display:inline-block;background:#eaf2fb;color:var(--accent);border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;margin-bottom:14px}
    @media(prefers-color-scheme:dark){
      :root{--ink:#e8e8e8;--muted:#a0a0a0;--line:#2a2f3a;--bg:#0f1218}
      body{background:var(--bg)} section,.kpi{background:#161a22}
      .insights{background:#0f1218} header{background:linear-gradient(135deg,#0b2545,#1a2b4a)}
      .fig img{background:#fff}
      nav{background:rgba(15,18,24,.92)}
    }
    """
    nav = "".join(
        f'<a href="#s{i}">{t.split("—")[0].strip()}</a>'
        for i, (t, *_ ) in enumerate(REPORT_SECTIONS))
    nav = '<a href="#top">Overview</a>' + nav

    kpis = [
        (f"{len(df):,}", "Incidents"),
        (f"{df['City'].nunique()}", "Cities"),
        (f"{df['Crime Type'].nunique()}", "Crime types"),
        (f"{df['ArrestFlag'].mean():.0%}", "Arrest rate"),
        (f"{(df['Severity']=='High').mean():.0%}", "High severity"),
        (f"{df['Date'].min().year}–{df['Date'].max().year}", "Period"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="n">{n}</div><div class="l">{l}</div></div>'
        for n, l in kpis)

    secs = []
    for i, (title, intro, imgs, insights) in enumerate(REPORT_SECTIONS):
        figs = "".join(
            f'<div class="fig"><img src="data:image/png;base64,{_b64(p)}" '
            f'alt="{c}"><div class="cap">{c}</div></div>'
            for p, c in imgs)
        ins = "".join(f"<li>{x}</li>" for x in insights)
        secs.append(f"""
        <section id="s{i}">
          <div class="badge">Task {i+1}</div>
          <h2>{title}</h2>
          <p class="intro">{intro}</p>
          {figs}
          <div class="insights"><h3>Key findings</h3><ul>{ins}</ul></div>
        </section>""")

    html = f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Crime Analysis — EDA & Cluster Analysis</title><style>{css}</style></head>
    <body><a id="top"></a>
    <header><div class="wrap">
      <h1>Crime Analysis Report</h1>
      <p>Exploratory Data Analysis &amp; Cluster Analysis · Course-End Project</p>
      <p>{len(df):,} incidents · {df['City'].nunique()} cities · """ + \
        f"""{df['Date'].min().date()} to {df['Date'].max().date()}</p>
    </div></header>
    <div class="wrap">
      <div class="kpis">{kpi_html}</div>
      <nav>{nav}</nav>
      {''.join(secs)}
      <footer>Generated with pandas, scikit-learn &amp; matplotlib · charts use a
      colour-blind-safe palette · all underlying tables are available as CSV and
      in the accompanying Excel workbook.</footer>
    </div></body></html>"""

    path = os.path.join(OUT, "Crime_Analysis_Report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  html   -> outputs/Crime_Analysis_Report.html")
    return path


# ============================================================================
def main():
    print("=" * 68)
    print("CRIME ANALYSIS — EDA & CLUSTER ANALYSIS")
    print("=" * 68)
    df = load()
    print(f"Loaded {len(df)} rows x {df.shape[1]} cols from {os.path.basename(DATA)}")

    geo = task1_overall(df)
    task2_time(df)
    task3_trend(df)
    task4_compare(df)
    task5_cluster(df, geo)

    build_excel(df)
    build_html(df)

    # Cleaned dataset export
    df.to_csv(os.path.join(OUT, "crime_data_enriched.csv"), index=False)
    print("  data   -> outputs/crime_data_enriched.csv")
    print("\nDone. All outputs in:", OUT)


if __name__ == "__main__":
    main()
