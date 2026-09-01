"""
================================================================================
STEP 3: EXPLORATORY DATA ANALYSIS (LAYMAN SCRIPT)
================================================================================
Parses dataset metadata, patient demographics, and generates 12-lead signal plots.
Usage:
  PYTHONPATH=. python3 src/03_exploratory_data_analysis.py
================================================================================
"""

from src.eda import run_eda

if __name__ == "__main__":
    run_eda()
