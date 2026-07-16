"""
01_data_preparation.py  -  Week 2 (Data Science), Part 1
========================================================
Tasks addressed
---------------
1. Read every raw file and APPEND the versions, producing two tidy datasets:
   one experiment dataset and one blade-geometry dataset.
2. Rename all variables to follow the Python identifier naming convention.
3. Convert the a-dimensional radius (r/R) and a-dimensional chord (c/R) into
   physical RADIUS and CHORD distributions by multiplying with the blade
   radius R = Diameter / 2 (for every blade of every propeller).

Data note: the experiment data ships in three volumes (Experiment_vol1..3) and
the geometry data in two volumes (Geom_vol1..2). We read whatever volume files
are present for each prefix, so the pipeline is robust to the number of parts.

Outputs (data/processed/):
   experiment_all.csv     - appended + renamed experiment data
   geometry_all.csv       - appended + renamed geometry data + radius/chord dist
"""
import glob
import numpy as np
import pandas as pd

from config import RAW, PROCESSED, to_identifier
import os


def load_and_append(prefix):
    """Read every <prefix>_vol*.csv present in the raw folder and append them."""
    paths = sorted(glob.glob(os.path.join(RAW, f"{prefix}_vol*.csv")))
    if not paths:
        raise FileNotFoundError(f"No raw files matching {prefix}_vol*.csv in {RAW}")
    frames = []
    for path in paths:
        vol = int(os.path.basename(path).split("vol")[1].split(".")[0])
        df = pd.read_csv(path)
        df["source_volume"] = vol          # keep provenance of each row
        frames.append(df)
        print(f"  read {os.path.basename(path)}  ->  {df.shape[0]} rows, {df.shape[1]-1} cols")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  appended -> {combined.shape[0]} rows")
    return combined


def rename_to_identifiers(df):
    """Rename every column to a valid Python identifier and report the map."""
    mapping = {c: to_identifier(c) for c in df.columns}
    print("  column rename map:")
    for k, v in mapping.items():
        print(f"    {k!r:45s} -> {v}")
    return df.rename(columns=mapping)


def add_physical_distributions(geom):
    """radius = (r/R) * R  and  chord = (c/R) * R, with R = diameter / 2."""
    R = geom["propeller_diameter"] / 2.0
    geom["blade_radius_R"] = R
    geom["radius_distribution"] = geom["adimensional_radius_r_r"] * R
    geom["chord_distribution"] = geom["adimensional_chord_c_r"] * R
    return geom


def main():
    print("STEP 1  Reading & appending EXPERIMENT workbooks")
    experiment = load_and_append("Experiment")
    print("\nSTEP 2  Renaming EXPERIMENT columns to Python identifiers")
    experiment = rename_to_identifiers(experiment)

    print("\nSTEP 1  Reading & appending GEOMETRY workbooks")
    geometry = load_and_append("Geom")
    print("\nSTEP 2  Renaming GEOMETRY columns to Python identifiers")
    geometry = rename_to_identifiers(geometry)

    print("\nSTEP 3  Building physical radius & chord distributions")
    geometry = add_physical_distributions(geometry)
    print(geometry[["blade_name", "adimensional_radius_r_r",
                    "adimensional_chord_c_r", "radius_distribution",
                    "chord_distribution"]].head(6).to_string(index=False))

    exp_out = os.path.join(PROCESSED, "experiment_all.csv")
    geo_out = os.path.join(PROCESSED, "geometry_all.csv")
    experiment.to_csv(exp_out, index=False)
    geometry.to_csv(geo_out, index=False)
    print(f"\nSaved {exp_out}  ({experiment.shape})")
    print(f"Saved {geo_out}  ({geometry.shape})")

    # Quick provenance summary
    print("\nEXPERIMENT unique propellers:", experiment["propeller_name"].nunique())
    print("GEOMETRY   unique blades    :", geometry["blade_name"].nunique())


if __name__ == "__main__":
    main()
