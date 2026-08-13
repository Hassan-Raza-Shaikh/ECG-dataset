import os
import numpy as np
import pandas as pd
import wfdb
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def _read_signal(fpath):
    try:
        sig, _ = wfdb.rdsamp(fpath)
        return sig.astype(np.float32)
    except Exception as e:
        return np.zeros((1000, 12), dtype=np.float32)

def build_cache(raw_dir="data/raw/ptbxl", database_csv="data/raw/ptbxl/ptbxl_database.csv", sampling_rate=100):
    cache_path = os.path.join(raw_dir, f"signals_cache_{sampling_rate}hz.npy")
    if os.path.exists(cache_path):
        print(f"Cache already exists at {cache_path}")
        return

    df = pd.read_csv(database_csv, index_col="ecg_id")
    fn_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    filepaths = [os.path.join(raw_dir, f) for f in df[fn_col]]

    print(f"Building cache for {len(filepaths)} ECG records using {os.cpu_count()} CPU cores...")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        signals = list(tqdm(executor.map(_read_signal, filepaths, chunksize=100), total=len(filepaths)))

    signals_array = np.array(signals, dtype=np.float32)
    print(f"Saving binary cache to {cache_path} (Shape: {signals_array.shape}, Size: {signals_array.nbytes / 1e6:.2f} MB)...")
    np.save(cache_path, signals_array)
    print("Cache creation complete!")

if __name__ == "__main__":
    build_cache()
