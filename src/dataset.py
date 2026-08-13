import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb

from src.preprocessing import ECGPreprocessor

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def compute_superclasses(df, scp_df):
    """
    Maps SCP diagnostic codes to 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP.
    """
    diag_scp_df = scp_df[scp_df.diagnostic == 1]
    code_to_superclass = dict(zip(diag_scp_df.index, diag_scp_df.diagnostic_class))

    def aggregate_diagnostic(scp_codes):
        classes = set()
        for code in scp_codes.keys():
            if code in code_to_superclass:
                sclass = code_to_superclass[code]
                if pd.notna(sclass):
                    classes.add(sclass)
        return list(classes)

    df["scp_codes_dict"] = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    df["superclasses"] = df.scp_codes_dict.apply(aggregate_diagnostic)

    for superclass in SUPERCLASSES:
        df[superclass] = df.superclasses.apply(lambda x: 1 if superclass in x else 0)

    return df


from concurrent.futures import ProcessPoolExecutor, as_completed

def _read_single_signal(fpath):
    sig, _ = wfdb.rdsamp(fpath)
    return sig.astype(np.float32)

def load_raw_data_cached(df, raw_dir, sampling_rate):
    """Loads all ECG signals into a single cached numpy array file using multi-processing."""
    cache_path = os.path.join(raw_dir, f"signals_cache_{sampling_rate}hz.npy")
    fn_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    filepaths = [os.path.join(raw_dir, f) for f in df[fn_col]]

    if os.path.exists(cache_path):
        print(f"Loading cached signal binary from {cache_path}...", flush=True)
        signals = np.load(cache_path)
    else:
        print(f"Reading WFDB signals in parallel across CPU cores ({len(filepaths)} files)...", flush=True)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            signals = list(executor.map(_read_single_signal, filepaths, chunksize=100))
        signals = np.array(signals, dtype=np.float32)
        print(f"Saving binary cache to {cache_path}...", flush=True)
        np.save(cache_path, signals)

    return signals


class ECGDataset(Dataset):
    """
    PyTorch Dataset for 12-lead ECG classification.
    """

    def __init__(self, signals, labels, preprocessor=None):
        """
        signals: numpy array of shape (N, time_steps, num_leads)
        labels: numpy array of shape (N, num_classes)
        preprocessor: instance of ECGPreprocessor
        """
        self.signals = signals
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx] # (time_steps, num_leads)

        if self.preprocessor is not None:
            sig = self.preprocessor.process(sig)

        # Transpose signal to (num_leads, time_steps) for PyTorch Conv1D layers
        sig_tensor = torch.tensor(sig.T, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sig_tensor, label_tensor


def get_dataloaders(config):
    """
    Loads dataset metadata, processes superclasses, splits into train/val/test,
    and returns PyTorch DataLoaders.
    """
    raw_dir = config["data"]["raw_dir"]
    db_path = config["data"]["database_csv"]
    scp_path = config["data"]["scp_csv"]
    sampling_rate = config["data"]["sampling_rate"]
    
    # Read metadata
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)

    # Compute binary superclasses and load all signals into cached matrix
    df = compute_superclasses(df, scp_df)

    print(f"Loading raw ECG signals ({sampling_rate} Hz)...", flush=True)
    all_signals = load_raw_data_cached(df, raw_dir, sampling_rate)

    # Split dataset based on strat_fold
    train_folds = config["split"]["train_folds"]
    val_folds = config["split"]["val_folds"]
    test_folds = config["split"]["test_folds"]

    train_idx = df.strat_fold.isin(train_folds).values
    val_idx = df.strat_fold.isin(val_folds).values
    test_idx = df.strat_fold.isin(test_folds).values

    print(f"Dataset split: Train={train_idx.sum()}, Val={val_idx.sum()}, Test={test_idx.sum()}", flush=True)

    X_train, y_train = all_signals[train_idx], df[SUPERCLASSES].values[train_idx]
    X_val, y_val = all_signals[val_idx], df[SUPERCLASSES].values[val_idx]
    X_test, y_test = all_signals[test_idx], df[SUPERCLASSES].values[test_idx]

    # Preprocessor
    preproc_config = config["preprocessing"]
    preprocessor = ECGPreprocessor(
        sampling_rate=sampling_rate,
        highpass_cutoff=preproc_config["highpass_cutoff"],
        lowpass_cutoff=preproc_config["lowpass_cutoff"],
        notch_freq=preproc_config["notch_freq"],
        notch_q=preproc_config["notch_q"],
        apply_wavelet=preproc_config["apply_wavelet_denoising"],
        wavelet_name=preproc_config["wavelet_name"],
        wavelet_level=preproc_config["wavelet_level"],
        normalize=preproc_config["normalize"],
    ) if preproc_config["apply_denoising"] else None

    # Datasets
    train_dataset = ECGDataset(X_train, y_train, preprocessor=preprocessor)
    val_dataset = ECGDataset(X_val, y_val, preprocessor=preprocessor)
    test_dataset = ECGDataset(X_test, y_test, preprocessor=preprocessor)

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, SUPERCLASSES

