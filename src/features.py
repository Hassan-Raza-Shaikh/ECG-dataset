import os
import numpy as np
import scipy.signal as signal
import pywt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def extract_lead_features(lead_sig, sampling_rate=100):
    """
    Extracts time-domain, spectral, and wavelet features for a single 1D ECG lead.
    
    Returns 16 scalar features per lead:
    - Time-domain (7): mean, std, var, skewness, kurtosis, peak-to-peak, RMS
    - Spectral (5): PSD band energies (0.5-4, 4-8, 8-15, 15-30, 30-40 Hz)
    - Wavelet (4): Energy across DWT decomposition detail levels (level 4 sym8)
    """
    feats = []

    # 1. Time-Domain Features
    mean_val = np.mean(lead_sig)
    std_val = np.std(lead_sig)
    var_val = np.var(lead_sig)
    
    # Skewness & Kurtosis
    if std_val > 1e-8:
        skew_val = np.mean(((lead_sig - mean_val) / std_val) ** 3)
        kurt_val = np.mean(((lead_sig - mean_val) / std_val) ** 4) - 3.0
    else:
        skew_val, kurt_val = 0.0, 0.0

    ptp_val = np.ptp(lead_sig)
    rms_val = np.sqrt(np.mean(lead_sig ** 2))

    feats.extend([mean_val, std_val, var_val, skew_val, kurt_val, ptp_val, rms_val])

    # 2. Frequency-Domain (Spectral PSD) Features
    freqs, psd = signal.welch(lead_sig, fs=sampling_rate, nperseg=min(256, len(lead_sig)))
    
    bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 30.0), (30.0, 40.0)]
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    for low, high in bands:
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_energy = trapz_fn(psd[idx_band], freqs[idx_band]) if np.any(idx_band) else 0.0
        feats.append(band_energy)

    # 3. Wavelet Energy Features (DWT sym8, level 4)
    try:
        coeffs = pywt.wavedec(lead_sig, "sym8", level=4)
        for c in coeffs[1:]: # detail coefficients
            feats.append(np.sum(c ** 2))
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def extract_ecg_record_features(signal_matrix, sampling_rate=100):
    """
    Extracts features across all 12 leads for a single ECG record (1000, 12).
    Returns 1D feature array of shape (12 * 16,) = 192 features.
    """
    record_feats = []
    num_leads = signal_matrix.shape[1]
    for lead_idx in range(num_leads):
        lead_feats = extract_lead_features(signal_matrix[:, lead_idx], sampling_rate=sampling_rate)
        record_feats.append(lead_feats)
    return np.concatenate(record_feats)


def _extract_record_worker(sig):
    return extract_ecg_record_features(sig)

def extract_dataset_features(signals_array, sampling_rate=100, raw_dir="data/raw/ptbxl"):
    """
    Extracts features for an entire dataset array (N, 1000, 12) with multi-processing.
    Caches extracted feature matrix as features_cache_100hz.npy.
    """
    cache_path = os.path.join(raw_dir, f"features_cache_{sampling_rate}hz.npy")
    if os.path.exists(cache_path):
        print(f"Loading cached extracted features from {cache_path}...", flush=True)
        return np.load(cache_path)

    print(f"Extracting 192 hand-crafted features for {len(signals_array)} ECG records across {os.cpu_count()} CPU cores...", flush=True)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        feature_matrix = list(tqdm(executor.map(_extract_record_worker, signals_array, chunksize=100), total=len(signals_array)))

    feature_matrix = np.array(feature_matrix, dtype=np.float32)
    print(f"Saving extracted feature matrix to {cache_path} (Shape: {feature_matrix.shape})...", flush=True)
    np.save(cache_path, feature_matrix)
    return feature_matrix
