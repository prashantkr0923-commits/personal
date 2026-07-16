"""Run the full propeller-performance pipeline end to end."""
import runpy
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "01_data_preparation.py",
    "02_solidity_analysis.py",
    "03_eda_visualizations.py",
    "04_ml_models.py",
    "05_run_sql.py",
    "06_tableau_dashboard.py",
]

for step in STEPS:
    print("\n" + "#" * 72 + f"\n# RUN {step}\n" + "#" * 72)
    runpy.run_path(os.path.join(HERE, step), run_name="__main__")

print("\nPipeline complete. See ../outputs/figures and ../outputs/tables.")
