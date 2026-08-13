import os
import ast
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb

from src.preprocessing import ECGPreprocessor

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def compute_superclasses(df, scp_df):
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


def plot_demographics(df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    # Age distribution
    sns.histplot(df["age"].dropna(), kde=True, ax=axes[0, 0], color="#2b5c8f")
    axes[0, 0].set_title("Patient Age Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Age (years)")

    # Sex distribution (0: Female, 1: Male in PTB-XL)
    sex_counts = df["sex"].value_counts().rename(index={0: "Female", 1: "Male"})
    axes[0, 1].pie(
        sex_counts,
        labels=sex_counts.index,
        autopct="%1.1f%%",
        colors=["#e76f51", "#457b9d"],
        startangle=140,
        textprops={"fontsize": 12},
    )
    axes[0, 1].set_title("Patient Sex Distribution", fontsize=14, fontweight="bold")

    # Height distribution
    sns.histplot(df["height"].dropna(), kde=True, ax=axes[1, 0], color="#2a9d8f")
    axes[1, 0].set_title("Patient Height Distribution", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Height (cm)")

    # Weight distribution
    sns.histplot(df["weight"].dropna(), kde=True, ax=axes[1, 1], color="#e76f51")
    axes[1, 1].set_title("Patient Weight Distribution", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Weight (kg)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved demographics plot to {save_path}")


def plot_superclass_distribution(df, save_path):
    counts = df[SUPERCLASSES].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=counts.index, y=counts.values, palette="crest")
    
    plt.title("Diagnostic Superclass Frequency Distribution (PTB-XL)", fontsize=14, fontweight="bold")
    plt.xlabel("Diagnostic Superclass", fontsize=12)
    plt.ylabel("Number of ECG Records", fontsize=12)

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved superclass distribution plot to {save_path}")


def plot_co_occurrence(df, save_path):
    labels_df = df[SUPERCLASSES]
    co_matrix = labels_df.T.dot(labels_df)

    plt.figure(figsize=(8, 6))
    sns.heatmap(co_matrix, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5)
    plt.title("Diagnostic Superclass Co-occurrence Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved co-occurrence matrix plot to {save_path}")


def plot_raw_vs_denoised(df, raw_dir, preprocessor, save_path):
    # Select first record
    sample_file = os.path.join(raw_dir, df.iloc[0].filename_lr)
    raw_signal, meta = wfdb.rdsamp(sample_file) # (1000, 12)
    lead_names = meta["sig_name"]

    denoised_signal = preprocessor.process(raw_signal)

    fig, axes = plt.subplots(12, 1, figsize=(14, 20), sharex=True)
    fig.suptitle("12-Lead ECG Signal: Raw vs. Filtered & Denoised", fontsize=16, fontweight="bold", y=0.995)

    time_axis = np.arange(raw_signal.shape[0]) / meta["fs"]

    for i in range(12):
        axes[i].plot(time_axis, raw_signal[:, i], label="Raw Signal", color="#e63946", alpha=0.6, linewidth=1.2)
        axes[i].plot(time_axis, denoised_signal[:, i], label="Filtered & Denoised", color="#1d3557", linewidth=1.5)
        axes[i].set_ylabel(f"Lead {lead_names[i]}\n(mV)", fontsize=10)
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved 12-lead raw vs denoised ECG plot to {save_path}")


def run_eda(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_dir = config["eda"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    db_path = config["data"]["database_csv"]
    scp_path = config["data"]["scp_csv"]
    raw_dir = config["data"]["raw_dir"]

    print("Loading PTB-XL database metadata...")
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)

    df = compute_superclasses(df, scp_df)

    print("\n--- PTB-XL Dataset Summary ---")
    print(f"Total Records: {len(df)}")
    print(f"Total Patients: {df.patient_id.nunique()}")
    print("\nSuperclass counts:")
    print(df[SUPERCLASSES].sum())

    # Plot Demographics
    plot_demographics(df, os.path.join(save_dir, "demographics.png"))

    # Plot Superclass distribution
    plot_superclass_distribution(df, os.path.join(save_dir, "superclass_distribution.png"))

    # Plot Co-occurrence
    plot_co_occurrence(df, os.path.join(save_dir, "co_occurrence.png"))

    # Denoising demonstration
    preproc_config = config["preprocessing"]
    preprocessor = ECGPreprocessor(
        sampling_rate=config["data"]["sampling_rate"],
        highpass_cutoff=preproc_config["highpass_cutoff"],
        lowpass_cutoff=preproc_config["lowpass_cutoff"],
        notch_freq=preproc_config["notch_freq"],
        notch_q=preproc_config["notch_q"],
        apply_wavelet=preproc_config["apply_wavelet_denoising"],
        wavelet_name=preproc_config["wavelet_name"],
        wavelet_level=preproc_config["wavelet_level"],
        normalize=preproc_config["normalize"],
    )

    plot_raw_vs_denoised(df, raw_dir, preprocessor, os.path.join(save_dir, "raw_vs_denoised_ecg.png"))

    print("\nEDA Completed successfully! All artifacts saved in:", save_dir)


if __name__ == "__main__":
    run_eda()
