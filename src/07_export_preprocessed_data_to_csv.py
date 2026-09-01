"""
================================================================================
STEP 7: EXPORT PREPROCESSED DATA & FEATURES TO CSV (LAYMAN SCRIPT)
================================================================================
Exports all patient metadata, superclasses, and 192 features into Excel/Pandas CSV:
`data/raw/ptbxl/preprocessed_features_100hz.csv`
Usage:
  PYTHONPATH=. python3 src/07_export_preprocessed_data_to_csv.py
================================================================================
"""

from src.export_preprocessed_csv import export_preprocessed_to_csv

if __name__ == "__main__":
    export_preprocessed_to_csv()
