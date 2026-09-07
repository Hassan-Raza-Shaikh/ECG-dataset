"""
================================================================================
🧠 STEP 3: NON-STATIONARY SIGNAL ANALYSIS & CHANNEL SELECTION
================================================================================
Adapting the Inter-Class vs. Intra-Class Variance method used in 19-channel EEG
channel selection to 12-channel ECG:
1. Computes local non-stationary temporal mean and variance across sliding windows.
2. Calculates Intra-Class Variance (S_W) within each diagnostic disease class.
3. Calculates Inter-Class Variance (S_B) between different disease classes.
4. Computes Fisher Discriminability Ratio (F = S_B / S_W) per channel.
5. Ranks all 12 channels and selects the Top 3 to 4 unique channels.
================================================================================
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset import compute_superclasses, SUPERCLASSES

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def run_channel_selection(config_path="configs/config.yaml", top_k=4):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    db_path = config["data"]["database_csv"]
    scp_path = config["data"]["scp_csv"]
    sampling_rate = config["data"]["sampling_rate"]

    print("=" * 70)
    print("STEP 3: NON-STATIONARY SIGNAL ANALYSIS & CHANNEL SELECTION")
    print("=" * 70)

    # 1. Load Metadata
    print(f"Loading metadata from {db_path}...")
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)
    df = compute_superclasses(df, scp_df)

    # Assign primary class for ANOVA/Fisher ratio
    primary_classes = []
    for idx, row in df.iterrows():
        matches = [c for c in SUPERCLASSES if row[c] == 1]
        primary_classes.append(matches[0] if matches else "NORM")
    labels = np.array(primary_classes)

    # 2. Load Signal Cache
    cache_path = os.path.join(raw_dir, f"signals_cache_{sampling_rate}hz.npy")
    if not os.path.exists(cache_path):
        # Fallback to 100 Hz if 500 Hz cache not built yet
        cache_path = os.path.join(raw_dir, "signals_cache_100hz.npy")
    
    if not os.path.exists(cache_path):
        print(f"Error: Signal cache not found at {cache_path}. Run benchmark first.")
        return

    print(f"Loading signal cache from {cache_path}...")
    signals = np.load(cache_path) # Shape: (N, TimeSteps, 12)
    print(f"Loaded signals shape: {signals.shape}")

    # 3. Non-Stationary Signal Analysis: Temporal Window Mean & Variance
    num_windows = 5
    windows = np.array_split(signals, num_windows, axis=1)
    win_vars = np.array([np.var(w, axis=1) for w in windows])   # (windows, N, 12)
    win_means = np.array([np.mean(w, axis=1) for w in windows]) # (windows, N, 12)

    # Non-stationary temporal variance metric (fluctuation of local energy & baseline)
    non_stat_feature = np.var(win_vars, axis=0) + np.var(win_means, axis=0) # (N, 12)

    # 4. Compute Inter-Class (S_B) and Intra-Class (S_W) Variance
    K = len(SUPERCLASSES)
    N_total = len(labels)
    channel_metrics = []

    for ch_idx, lead in enumerate(LEAD_NAMES):
        f_ch = non_stat_feature[:, ch_idx]
        global_mean = np.mean(f_ch)

        sb = 0.0 # Between-class variance
        sw = 0.0 # Within-class variance

        for c in SUPERCLASSES:
            mask = (labels == c)
            n_c = np.sum(mask)
            if n_c > 0:
                c_vals = f_ch[mask]
                c_mean = np.mean(c_vals)
                sb += n_c * ((c_mean - global_mean) ** 2)
                sw += np.sum((c_vals - c_mean) ** 2)

        sb = sb / (K - 1)
        sw = sw / (N_total - K)
        f_ratio = sb / (sw + 1e-10)

        channel_metrics.append({
            "Lead": lead,
            "Channel_Index": ch_idx,
            "Inter_Class_Var_SB": sb,
            "Intra_Class_Var_SW": sw,
            "Discriminability_Ratio": f_ratio
        })

    results_df = pd.DataFrame(channel_metrics).sort_values(by="Discriminability_Ratio", ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df.index.name = "Rank"

    print("\n" + "=" * 70)
    print("CHANNEL DISCRIMINABILITY RANKING (INTER-CLASS / INTRA-CLASS RATIO)")
    print("=" * 70)
    print(results_df.to_string())

    top_leads = results_df.head(top_k)["Lead"].tolist()
    top_indices = results_df.head(top_k)["Channel_Index"].tolist()

    print("\n" + "=" * 70)
    print(f"SELECTED TOP {top_k} UNIQUE CHANNELS: {top_leads}")
    print(f"Channel Lead Indices: {top_indices}")
    print("=" * 70)

    # 5. Visual Plot 1: Channel Ranking Bar Chart
    plt.figure(figsize=(12, 6))
    colors = ["#2a9d8f" if l in top_leads else "#a8dadc" for l in results_df["Lead"]]
    bars = plt.bar(results_df["Lead"], results_df["Discriminability_Ratio"], color=colors, edgecolor="#1d3557", linewidth=1.2)

    thresh = results_df.iloc[top_k - 1]["Discriminability_Ratio"]
    plt.axhline(thresh, color="#e63946", linestyle="--", linewidth=1.5, label=f"Selection Threshold (Top {top_k})")
    plt.title("ECG Channel Selection via Inter/Intra Class Discriminability Ratio (Fisher Criterion)", fontsize=13, fontweight="bold")
    plt.xlabel("ECG Lead (Channel)", fontsize=11, fontweight="bold")
    plt.ylabel("Discriminability Ratio (SB / SW)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plot1_path = "artifacts/eda/channel_selection_ranking.png"
    os.makedirs(os.path.dirname(plot1_path), exist_ok=True)
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved channel selection ranking plot to: {plot1_path}")

    # 6. Visual Plot 2: Waveform of the Selected 4 Channels
    sample_record = signals[0]
    time_axis = np.arange(len(sample_record)) / (500 if len(sample_record) == 5000 else 100)

    fig, axes = plt.subplots(top_k, 1, figsize=(14, 2.5 * top_k), sharex=True)
    fig.suptitle(f"Selected Top {top_k} Unique ECG Channels for Classification: {top_leads}", fontsize=14, fontweight="bold")

    for ax, lead_name, ch_idx in zip(axes, top_leads, top_indices):
        ax.plot(time_axis, sample_record[:, ch_idx], color="#1d3557", linewidth=1.3)
        ax.set_ylabel(f"{lead_name}\n(mV)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plot2_path = "artifacts/eda/selected_channels_ecg.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved selected channels waveform plot to: {plot2_path}")

    return top_leads, top_indices, results_df


if __name__ == "__main__":
    run_channel_selection(top_k=4)
