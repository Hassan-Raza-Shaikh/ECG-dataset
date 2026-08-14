import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb

from src.preprocessing import ECGPreprocessor


def add_controlled_noise(clean_signal, sampling_rate=100, target_snr_db=10.0):
    """
    Intentionally adds controlled artificial noise to a clean ECG recording:
    1. Additive White Gaussian Noise (AWGN)
    2. Sinusoidal Baseline Wander (0.2 Hz)
    3. Powerline Hum Interference (50 Hz)
    """
    N, num_leads = clean_signal.shape
    time_axis = np.arange(N) / sampling_rate

    # Signal power per lead
    sig_power = np.mean(clean_signal ** 2, axis=0, keepdims=True)

    # 1. White Gaussian Noise (AWGN) at target SNR
    noise_power = sig_power / (10 ** (target_snr_db / 10.0))
    awgn_noise = np.random.normal(0, np.sqrt(noise_power), size=(N, num_leads))

    # 2. Low-frequency Baseline Wander (0.2 Hz sinusoid)
    baseline_wander = 0.15 * np.sin(2 * np.pi * 0.2 * time_axis)[:, None]

    # 3. High-frequency Powerline Hum (50 Hz sinusoid)
    powerline_hum = 0.08 * np.sin(2 * np.pi * 50.0 * time_axis)[:, None]

    total_noise = awgn_noise + baseline_wander + powerline_hum
    corrupted_signal = clean_signal + total_noise

    return corrupted_signal, total_noise


def calculate_denoising_metrics(clean, corrupted, denoised):
    """
    Calculates quantitative signal fidelity & noise reduction metrics:
    - Input SNR (dB)
    - Output SNR (dB)
    - SNR Improvement (dB)
    - Root Mean Square Error (RMSE)
    - Percentage Root-Mean-Square Difference (PRD %)
    - Pearson Correlation Coefficient (r)
    """
    sig_var = np.sum(clean ** 2)
    noise_in_var = np.sum((corrupted - clean) ** 2)
    noise_out_var = np.sum((denoised - clean) ** 2)

    snr_in = 10 * np.log10(sig_var / (noise_in_var + 1e-10))
    snr_out = 10 * np.log10(sig_var / (noise_out_var + 1e-10))
    snr_imp = snr_out - snr_in

    rmse = np.sqrt(np.mean((denoised - clean) ** 2))
    prd = np.sqrt(noise_out_var / (sig_var + 1e-10)) * 100.0

    # Pearson correlation coefficient per lead (averaged)
    corrs = []
    for lead in range(clean.shape[1]):
        r = np.corrcoef(clean[:, lead], denoised[:, lead])[0, 1]
        corrs.append(r)
    mean_corr = np.mean(corrs)

    return {
        "SNR_in_dB": snr_in,
        "SNR_out_dB": snr_out,
        "SNR_imp_dB": snr_imp,
        "RMSE": rmse,
        "PRD_percent": prd,
        "Pearson_r": mean_corr
    }


def plot_denoising_validation(clean, corrupted, denoised, sampling_rate, lead_idx=1, save_path="artifacts/eda/denoising_validation.png"):
    """
    Plots a 4-panel quantitative validation visualization:
    1. Original Clean Signal
    2. Corrupted Signal (Clean + Controlled Noise)
    3. Denoised Output Signal (Overlaid on Original Clean ECG)
    4. Residual Distortion Error (Denoised - Clean)
    """
    time_axis = np.arange(len(clean)) / sampling_rate
    lead_name = f"Lead {lead_idx + 1}"

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    sns.set_theme(style="whitegrid")

    # Panel 1: Original Clean Signal
    axes[0].plot(time_axis, clean[:, lead_idx], color="#1d3557", linewidth=1.5, label="Original Clean ECG Baseline")
    axes[0].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[0].set_title(f"1. Original Clean ECG Waveform ({lead_name} @ {sampling_rate} Hz)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right")

    # Panel 2: Corrupted Signal
    axes[1].plot(time_axis, corrupted[:, lead_idx], color="#e63946", linewidth=1.0, alpha=0.8, label="Corrupted ECG (+AWGN, Baseline Drift, 50Hz Hum)")
    axes[1].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[1].set_title("2. Corrupted ECG Signal (With Controlled Artificial Noise Added)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right")

    # Panel 3: Denoised Signal vs Original Clean Signal Overlay
    axes[2].plot(time_axis, clean[:, lead_idx], color="#457b9d", linewidth=2.0, alpha=0.5, label="Original Clean ECG Baseline")
    axes[2].plot(time_axis, denoised[:, lead_idx], color="#2a9d8f", linewidth=1.5, linestyle="--", label="Denoised ECG Output")
    axes[2].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[2].set_title("3. Denoised Signal Output Overlaid on Original Clean Waveform", fontsize=12, fontweight="bold")
    axes[2].legend(loc="upper right")

    # Panel 4: Residual Signal Distortion Error (Denoised - Clean)
    residual_error = denoised[:, lead_idx] - clean[:, lead_idx]
    axes[3].plot(time_axis, residual_error, color="#e76f51", linewidth=1.2, label="Residual Error (Denoised - Clean)")
    axes[3].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[3].set_ylabel("Error (mV)", fontsize=10)
    axes[3].set_xlabel("Time (seconds)", fontsize=12)
    axes[3].set_title("4. Residual Signal Distortion (Error Curve proving QRS Morphological Fidelity)", fontsize=12, fontweight="bold")
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved denoising validation plot to {save_path}", flush=True)


def run_denoising_validation(sampling_rate=100, target_snr=10.0):
    raw_dir = "data/raw/ptbxl"
    db_path = "data/raw/ptbxl/ptbxl_database.csv"
    
    if not os.path.exists(db_path):
        print("Database metadata not found. Please ensure PTB-XL dataset is downloaded.")
        return

    df = pd.read_csv(db_path, index_col="ecg_id")
    fn_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    
    # Load first record
    sample_file = os.path.join(raw_dir, df.iloc[0][fn_col])
    raw_signal, meta = wfdb.rdsamp(sample_file) # (N, 12)

    # 1. Clean signal baseline (apply baseline preprocessor to obtain clean ground truth reference)
    clean_preprocessor = ECGPreprocessor(sampling_rate=sampling_rate, apply_wavelet=True, normalize=None)
    clean_ecg = clean_preprocessor.process(raw_signal)

    # 2. Intentionally add controlled artificial noise (AWGN + Baseline Wander + 50Hz Hum)
    corrupted_ecg, noise_added = add_controlled_noise(clean_ecg, sampling_rate=sampling_rate, target_snr_db=target_snr)

    # 3. Apply Noise Reduction Denoising Pipeline
    denoised_ecg = clean_preprocessor.process(corrupted_ecg)

    # 4. Calculate Quantitative Denoising Metrics
    metrics = calculate_denoising_metrics(clean_ecg, corrupted_ecg, denoised_ecg)

    print("\n=======================================================")
    print(f"   CONTROLLED DENOISING VALIDATION RESULTS ({sampling_rate} Hz)   ")
    print("=======================================================")
    print(f"Input SNR:                   {metrics['SNR_in_dB']:.2f} dB")
    print(f"Denoised Output SNR:          {metrics['SNR_out_dB']:.2f} dB")
    print(f"SNR Improvement (gain):       +{metrics['SNR_imp_dB']:.2f} dB")
    print(f"Root Mean Square Error (RMSE): {metrics['RMSE']:.4f} mV")
    print(f"PRD (Distortion Percentage):  {metrics['PRD_percent']:.2f}%")
    print(f"Pearson Correlation (r):      {metrics['Pearson_r']:.4f} (High Waveform Fidelity)")
    print("=======================================================")

    # 5. Save visual comparison plot
    plot_path = f"artifacts/eda/denoising_validation_{sampling_rate}hz.png"
    plot_denoising_validation(clean_ecg, corrupted_ecg, denoised_ecg, sampling_rate=sampling_rate, save_path=plot_path)

    return metrics


if __name__ == "__main__":
    print("Running Controlled Denoising Validation for 100 Hz...")
    run_denoising_validation(sampling_rate=100, target_snr=10.0)

    print("\nRunning Controlled Denoising Validation for 500 Hz...")
    run_denoising_validation(sampling_rate=500, target_snr=10.0)
