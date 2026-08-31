"""
================================================================================
🫀 STANDALONE 12-LEAD ECG DENOISING & PREPROCESSING DEMO (500 Hz)
================================================================================
This self-contained script demonstrates step-by-step ECG signal denoising for 
3 clinical records across all 12 leads sampled at 500 Hz (5,000 time-steps).

Filtering Pipeline applied to each 12-lead signal:
  1. High-Pass Butterworth Filter (0.5 Hz)  -> Removes Baseline Wander (breathing/motion)
  2. Low-Pass Butterworth Filter (40 Hz)   -> Removes EMG Muscle Artifacts
  3. IIR Notch Filter (50 Hz)               -> Removes Powerline Hum
  4. Discrete Wavelet Denoising (DWT sym8)  -> Removes Random High-Frequency Noise
  5. Per-Lead Z-Score Normalization         -> Zero-Mean Unit-Variance Scaling
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
import wfdb

# ==============================================================================
# 1. ECG PREPROCESSOR CLASS (500 Hz Configured)
# ==============================================================================
class StandaloneECGPreprocessor500Hz:
    def __init__(
        self,
        sampling_rate=500,        # 500 samples per second
        highpass_cutoff=0.5,      # Hz
        lowpass_cutoff=40.0,      # Hz
        notch_freq=50.0,          # Hz
        notch_q=30.0,             # Quality factor
        apply_wavelet=True,
        wavelet_name="sym8",
        wavelet_level=5,          # Level 5 for 500 Hz (since 500 / 2^5 = 15.6 Hz)
        normalize="zscore"
    ):
        self.fs = sampling_rate
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.apply_wavelet = apply_wavelet
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.normalize_method = normalize

    def remove_baseline_wander(self, sig):
        """High-pass Butterworth filter at 0.5 Hz"""
        nyquist = 0.5 * self.fs
        b, a = signal.butter(5, self.highpass_cutoff / nyquist, btype="highpass")
        return signal.filtfilt(b, a, sig, axis=0)

    def remove_muscle_noise(self, sig):
        """Low-pass Butterworth filter at 40 Hz"""
        nyquist = 0.5 * self.fs
        cutoff = min(self.lowpass_cutoff / nyquist, 0.99)
        b, a = signal.butter(5, cutoff, btype="lowpass")
        return signal.filtfilt(b, a, sig, axis=0)

    def remove_powerline_hum(self, sig):
        """IIR Notch filter at 50 Hz"""
        nyquist = 0.5 * self.fs
        if self.notch_freq >= nyquist:
            return sig
        b, a = signal.iirnotch(self.notch_freq, self.notch_q, fs=self.fs)
        return signal.filtfilt(b, a, sig, axis=0)

    def wavelet_denoise_lead(self, lead_sig):
        """DWT soft-thresholding per lead"""
        try:
            coeffs = pywt.wavedec(lead_sig, self.wavelet_name, level=self.wavelet_level)
            sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
            if sigma > 1e-8:
                uthresh = sigma * np.sqrt(2 * np.log(len(lead_sig)))
                new_coeffs = [coeffs[0]] + [
                    pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]
                ]
                denoised = pywt.waverec(new_coeffs, self.wavelet_name)
                return denoised[: len(lead_sig)].astype(np.float32)
            return lead_sig
        except Exception:
            return lead_sig

    def apply_wavelet_denoising(self, sig):
        """Applies DWT denoising to all 12 leads"""
        denoised = np.zeros_like(sig)
        for lead_idx in range(sig.shape[1]):
            denoised[:, lead_idx] = self.wavelet_denoise_lead(sig[:, lead_idx])
        return denoised

    def zscore_normalize(self, sig):
        """Z-score normalization per lead"""
        mean = np.mean(sig, axis=0, keepdims=True)
        std = np.std(sig, axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (sig - mean) / std

    def process(self, raw_signal):
        """
        Input:  raw_signal of shape (5000, 12)  [5000 time-steps x 12 leads]
        Output: cleaned_signal of shape (5000, 12)
        """
        sig = raw_signal.copy().astype(np.float32)

        # Step 1: Baseline wander removal (0.5 Hz highpass)
        sig = self.remove_baseline_wander(sig)

        # Step 2: EMG muscle artifact removal (40 Hz lowpass)
        sig = self.remove_muscle_noise(sig)

        # Step 3: Powerline 50 Hz hum removal (Notch filter)
        sig = self.remove_powerline_hum(sig)

        # Step 4: Discrete Wavelet Transform soft-thresholding
        if self.apply_wavelet:
            sig = self.apply_wavelet_denoising(sig)

        # Step 5: Per-lead Z-score scaling
        if self.normalize_method == "zscore":
            sig = self.zscore_normalize(sig)

        return sig


# ==============================================================================
# 2. PLOTTING FUNCTION FOR ALL 12 LEADS
# ==============================================================================
def plot_12_lead_ecg(raw_sig, clean_sig, lead_names, record_id, save_path, fs=500):
    """
    Plots all 12 leads of an ECG record (Raw vs. Denoised) side-by-side / overlay.
    """
    time_axis = np.arange(len(raw_sig)) / fs # 10 seconds (0 to 10s)

    fig, axes = plt.subplots(12, 1, figsize=(15, 22), sharex=True)
    fig.suptitle(
        f"12-Lead ECG Denoising Demonstration (Record #{record_id} @ {fs} Hz)\n"
        f"Red: Raw Corrupted Signal | Blue/Navy: Filtered & Denoised Output",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )

    for i in range(12):
        lead_label = lead_names[i] if i < len(lead_names) else f"Lead {i+1}"
        axes[i].plot(time_axis, raw_sig[:, i], label="Raw Signal", color="#e63946", alpha=0.5, linewidth=1.0)
        axes[i].plot(time_axis, clean_sig[:, i], label="Denoised Signal", color="#1d3557", linewidth=1.4)
        axes[i].set_ylabel(f"{lead_label}\n(mV)", fontsize=9, fontweight="bold")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved 12-lead plot for Record #{record_id} to: {save_path}")


# ==============================================================================
# 3. MAIN EXECUTION (Processes 3 Records @ 500 Hz)
# ==============================================================================
def main():
    raw_dir = "data/raw/ptbxl"
    db_path = "data/raw/ptbxl/ptbxl_database.csv"

    if not os.path.exists(db_path):
        print(f"Error: Dataset metadata not found at {db_path}.")
        return

    # Load PTB-XL metadata CSV
    df = pd.read_csv(db_path, index_col="ecg_id")
    print(f"Loaded PTB-XL metadata ({len(df)} total records).")

    # Instantiate 500 Hz Preprocessor
    preprocessor = StandaloneECGPreprocessor500Hz(sampling_rate=500)

    # Select 3 records to process
    record_ids = df.index[:3] # First 3 records in dataset

    print("\n" + "=" * 70)
    print("STARTING 12-LEAD ECG DENOISING DEMO FOR 3 RECORDS (500 Hz)")
    print("=" * 70)

    for idx, ecg_id in enumerate(record_ids, 1):
        row = df.loc[ecg_id]
        fn_500hz = row["filename_hr"] # 500 Hz high-resolution file path e.g. records500/00000/00001_hr
        full_path = os.path.join(raw_dir, fn_500hz)

        print(f"\nProcessing Record {idx}/3 [ECG ID #{ecg_id}]...")
        print(f"  File Path: {full_path}.dat")

        # Load 500 Hz WFDB raw signal
        raw_signal, meta = wfdb.rdsamp(full_path) # Shape: (5000, 12)
        lead_names = meta["sig_name"] # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        sampling_rate = meta["fs"]

        print(f"  Loaded Signal Shape: {raw_signal.shape} ({sampling_rate} Hz, {meta['sig_len']} samples per lead)")
        print(f"  Leads: {', '.join(lead_names)}")

        # Execute Denoising Pipeline
        denoised_signal = preprocessor.process(raw_signal)

        # Plot & Save all 12 Leads
        save_plot_path = f"artifacts/eda/demo_500hz_record_{ecg_id}.png"
        plot_12_lead_ecg(raw_signal, denoised_signal, lead_names, ecg_id, save_plot_path, fs=sampling_rate)

    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("Check generated 12-lead visual plots in: artifacts/eda/")
    print("=" * 70)


if __name__ == "__main__":
    main()
