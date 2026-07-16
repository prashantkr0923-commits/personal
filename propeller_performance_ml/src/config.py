"""Shared paths and helpers for the propeller-performance pipeline."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
PROCESSED = os.path.join(HERE, "..", "data", "processed")
FIGURES = os.path.join(HERE, "..", "outputs", "figures")
TABLES = os.path.join(HERE, "..", "outputs", "tables")

for _d in (PROCESSED, FIGURES, TABLES):
    os.makedirs(_d, exist_ok=True)


def to_identifier(name: str) -> str:
    """Convert a human column label into a valid, PEP 8-style Python identifier.

    'Propeller's Name'                 -> 'propeller_name'
    'Adimensional Chord - c/R'         -> 'adimensional_chord_c_r'
    'beta - Angle Relative to Rotation'-> 'beta_angle_relative_to_rotation'
    """
    s = name.strip().lower()
    s = s.replace("'s", "")            # drop possessive 's
    s = re.sub(r"[^0-9a-z]+", "_", s)  # non-alphanumerics -> underscore
    s = re.sub(r"_+", "_", s).strip("_")
    if s and s[0].isdigit():
        s = "_" + s
    return s
