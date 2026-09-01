"""
================================================================================
📄 EXPORT PREPROCESSED DATASET & FEATURES TO CSV
================================================================================
Exports the 192 preprocessed biological features (time-domain, spectral PSD, 
wavelet energies) and diagnostic superclasses into a clean CSV file:
`data/raw/ptbxl/preprocessed_features.csv`
================================================================================
"""

import os
import yaml
import numpy as np
import pandas as pd

from src.dataset import compute_superclasses, SUPERCLASSES
from src.features import extract_dataset_features


def export_preprocessed_to_csv(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    db_path = config["data"]["database_csv"]
    scp_path = config["data"]["scp_csv"]
    sampling_rate = config["data"]["sampling_rate"]

    output_csv = os.path.join(raw_dir, f"preprocessed_features_{sampling_rate}hz.csv")

    # 1. Load Metadata
    print(f"Loading metadata from {db_path}...")
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)
    df = compute_superclasses(df, scp_df)

    # 2. Load Signal Cache
    cache_path = os.path.join(raw_dir, f"signals_cache_{sampling_rate}hz.npy")
    if not os.path.exists(cache_path):
        print(f"Signal cache not found at {cache_path}. Run benchmark.py first.")
        return

    all_signals = np.load(cache_path)

    # 3. Extract or Load 192 Features
    print(f"Loading 192 preprocessed features ({sampling_rate} Hz)...")
    X_features = extract_dataset_features(all_signals, sampling_rate=sampling_rate, raw_dir=raw_dir)

    # 4. Construct DataFrame
    feat_cols = [f"feature_{i+1}" for i in range(X_features.shape[1])]
    features_df = pd.DataFrame(X_features, index=df.index, columns=feat_cols)

    # Combine metadata, superclass labels, and preprocessed features
    meta_cols = ["patient_id", "age", "sex", "height", "weight", "strat_fold"]
    combined_df = pd.concat([df[meta_cols + SUPERCLASSES], features_df], axis=1)

    print(f"Saving preprocessed CSV to {output_csv} (Shape: {combined_df.shape})...")
    combined_df.to_csv(output_csv)
    print(f"Export complete! File created at: {output_csv}")


if __name__ == "__main__":
    export_preprocessed_to_csv()
