"""
================================================================================
STEP 1: DOWNLOAD DATASET (LAYMAN SCRIPT)
================================================================================
Automatically downloads and extracts the PTB-XL ECG dataset (v1.0.3) from PhysioNet.
Usage:
  python src/01_download_dataset.py
================================================================================
"""

from src.download_dataset import download_and_extract_ptbxl

if __name__ == "__main__":
    download_and_extract_ptbxl()
