"""
run_all.py - execute every task script in order
===============================================

Convenience runner that executes Tasks 1 through 8 sequentially so the whole
analysis (console output + all charts under outputs/) can be reproduced with a
single command:

    python run_all.py
"""

import runpy

TASKS = [
    "task1_data_import_and_examination",
    "task2_missing_values_and_data_cleaning",
    "task3_feature_engineering",
    "task4_outlier_detection_and_treatment",
    "task5_categorical_encoding",
    "task6_correlation_heatmap",
    "task7_hypothesis_testing",
    "task8_business_visualizations",
]

if __name__ == "__main__":
    for mod in TASKS:
        print("\n\n" + "#" * 72)
        print("#  RUNNING:", mod)
        print("#" * 72)
        runpy.run_module(mod, run_name="__main__")
