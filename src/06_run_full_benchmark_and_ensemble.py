"""
================================================================================
STEP 6: RUN FULL BENCHMARK & STACKING ENSEMBLE (LAYMAN SCRIPT)
================================================================================
Runs end-to-end evaluation across all 8 models & constructs the Stacking Ensemble.
Usage:
  PYTHONPATH=. python3 src/06_run_full_benchmark_and_ensemble.py
================================================================================
"""

from src.benchmark import run_full_benchmark

if __name__ == "__main__":
    run_full_benchmark()
