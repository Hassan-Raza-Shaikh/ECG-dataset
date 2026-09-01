# 🫀 PTB-XL 12-Lead ECG Machine Learning & Signal Processing Pipeline

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Dataset](https://img.shields.io/badge/Dataset-PTB--XL--v1.0.3-green.svg)
![Open Source](https://img.shields.io/badge/Open_Source-Public-brightgreen.svg)

A production-grade, state-of-the-art end-to-end Machine Learning, Signal Denoising, and Multi-Model Benchmarking pipeline built for the **PTB-XL 12-lead Electrocardiography Dataset** (21,837 clinical 10-second ECG records from 18,885 patients).

---

## 🗺️ Simple & Clear File Directory Guide (Step-by-Step)

If you are new to the codebase, every script in [`src/`](src/) is numbered sequentially so anyone can follow the workflow step-by-step:

| Number / Script File | Plain English Explanation (What it Does) | Command to Run |
| :--- | :--- | :--- |
| 1️⃣ **`demo_denoising.py`** | **Single-File Denoising Demo**: Self-contained demo script showing 12-lead filtering for 3 records at 500 Hz. | `python demo_denoising.py` |
| 2️⃣ **`src/01_download_dataset.py`** | **Step 1 - Dataset Downloader**: Auto-downloads PTB-XL (v1.0.3) from PhysioNet directly into `data/raw/ptbxl/`. | `python src/01_download_dataset.py` |
| 3️⃣ **`src/02_verify_denoising_quality.py`** | **Step 2 - Noise Removal Verification**: Proves quantitatively (+16 dB SNR Gain, 97.4% $r$) that noise is removed without distorting ECG peaks. | `PYTHONPATH=. python3 src/02_verify_denoising_quality.py` |
| 4️⃣ **`src/03_exploratory_data_analysis.py`** | **Step 3 - Data Visualizer (EDA)**: Analyzes demographics, diagnostic frequencies, and generates 12-lead signal plots. | `PYTHONPATH=. python3 src/03_exploratory_data_analysis.py` |
| 5️⃣ **`src/04_train_classical_ml_models.py`** | **Step 4 - Classical ML Models**: Extracts 192 features and trains Random Forest, HistGradientBoosting, ExtraTrees, SVM, and Logistic Regression. | `PYTHONPATH=. python3 src/04_train_classical_ml_models.py` |
| 6️⃣ **`src/05_train_deep_learning_resnet.py`** | **Step 5 - Deep Neural Net (ResNet1D)**: Trains the 1D Deep Residual CNN architecture on Apple Silicon GPU/MPS or CUDA. | `PYTHONPATH=. python3 src/05_train_deep_learning_resnet.py` |
| 7️⃣ **`src/06_run_full_benchmark_and_ensemble.py`**| **Step 6 - Full Benchmark & Ensemble**: Compares all 8 models and evaluates the **Weighted Stacking Ensemble (91.82% AUC Champion)**. | `PYTHONPATH=. python3 src/06_run_full_benchmark_and_ensemble.py` |
| 8️⃣ **`src/07_export_preprocessed_data_to_csv.py`** | **Step 7 - Export Features to CSV**: Exports metadata, superclasses, and 192 preprocessed features into Excel/Pandas CSV. | `PYTHONPATH=. python3 src/07_export_preprocessed_data_to_csv.py` |

---

## 🏆 Multi-Model Benchmark Leaderboard (Held-Out Test Set Fold 10)

| Rank | Model Architecture / Ensemble | Model Type | Test Macro ROC-AUC | Test Precision | Test Macro F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Weighted Stacking Ensemble** | **Hybrid Deep+Tree Stacking** | **0.9182 (91.82%)** | **0.8030** | **0.6759** |
| 🥈 **2** | **ResNet1D** | Deep Residual 1D CNN | **0.9079 (90.79%)** | 0.7663 | 0.6696 |
| 🥉 **3** | **CNN-BiLSTM-Attention** | Spatial-Temporal Deep Net | **0.9028 (90.28%)** | 0.7591 | **0.6761** |
| 4 | **HistGradientBoosting** | Histogram Tree Ensemble | **0.8675 (86.75%)** | 0.7474 | 0.6033 |
| 5 | **RandomForest** | Random Decision Forests | **0.8512 (85.12%)** | 0.7621 | 0.5431 |
| 6 | **ExtraTrees** | Extremely Randomized Trees | **0.8274 (82.74%)** | 0.7704 | 0.4577 |
| 7 | **SVM_Linear** | Calibrated Support Vector Machine | **0.8263 (82.63%)** | 0.7310 | 0.5179 |
| 8 | **LogisticRegression** | L2 Regularized Linear Model | **0.8240 (82.40%)** | 0.7147 | 0.5381 |

For detailed model performance reports and ROC curves, see [`docs/MODEL_BENCHMARK.md`](docs/MODEL_BENCHMARK.md).

---

## 🎛️ Signal Processing & Denoising Architecture

Clinical ECG signals suffer from baseline wander, powerline hum, and muscle noise. Our filtering pipeline includes:

1. **Baseline Wander Removal (High-Pass Filter)**: 5th-order zero-phase Butterworth filter at 0.5 Hz cutoff frequency.
2. **EMG Muscle Noise Removal (Low-Pass Filter)**: 5th-order Butterworth filter at 40 Hz cutoff frequency.
3. **Powerline Hum Removal (Notch Filter)**: IIR Notch filter targeting 50 Hz AC electrical grid hum.
4. **Wavelet Denoising (DWT)**: Discrete Wavelet Transform using `sym8` wavelets with Donoho-Johnstone soft-thresholding.
5. **Per-Lead Normalization**: Zero-mean, unit-variance Z-score normalization per lead.

For full mathematical documentation, see [`docs/SIGNAL_DENOISING.md`](docs/SIGNAL_DENOISING.md).

---

## 📁 Repository Structure

```
ECG-dataset/
├── demo_denoising.py                  # Standalone 1-file demo script (500 Hz 12-lead filtering)
├── data/
│   └── raw/
│       └── ptbxl/                     # Cleaned PTB-XL dataset & binary cache files
├── src/
│   ├── 01_download_dataset.py         # Step 1: Auto-downloader for PTB-XL from PhysioNet
│   ├── 02_verify_denoising_quality.py # Step 2: Controlled noise validation (+16 dB SNR Gain)
│   ├── 03_exploratory_data_analysis.py# Step 3: EDA Visualizer & 12-lead signal plotter
│   ├── 04_train_classical_ml_models.py# Step 4: Train RandomForest, HistGradientBoosting, SVM
│   ├── 05_train_deep_learning_resnet.py# Step 5: Train Deep ResNet1D Neural Network
│   ├── 06_run_full_benchmark_and_ensemble.py # Step 6: Run full benchmark & Stacking Ensemble
│   ├── 07_export_preprocessed_data_to_csv.py # Step 7: Export 192 features & metadata to CSV
│   ├── preprocessing.py               # Core ECG Denoising Module (Butterworth, Notch, Wavelet)
│   ├── dataset.py                     # PyTorch Dataset parser & fast DataLoader
│   ├── features.py                    # 192 Biological Feature Extractor
│   ├── models.py                      # ResNet1D & CNN-BiLSTM-Attention Neural Networks
│   ├── train.py                       # ResNet1D training loop
│   ├── train_traditional.py           # Traditional ML trainer
│   ├── benchmark.py                   # Benchmark & Stacking Ensemble builder
│   ├── validate_denoising.py          # Controlled Noise Verification Engine
│   └── export_preprocessed_csv.py     # Preprocessed CSV exporter
├── docs/
│   ├── SIGNAL_DENOISING.md            # Technical guide on ECG signal preprocessing & filtering math
│   ├── EDA_REPORT.md                  # Comprehensive Exploratory Data Analysis report
│   └── MODEL_BENCHMARK.md             # Detailed Multi-Model & Stacking Ensemble benchmark leaderboard
├── configs/
│   └── config.yaml                    # Hyperparameters, filter cutoffs, and pipeline paths
├── artifacts/
│   ├── eda/                           # Saved EDA plots (demographics, class distribution, raw vs denoised ECG)
│   └── models/                        # Saved best model checkpoints, ROC curves, and training metrics
├── .gitignore                         # Git ignore rules for dataset binaries & model checkpoints
├── requirements.txt                   # Python package dependencies
└── README.md                          # Master project guide & documentation
```

---

## 📚 Technical Documentation Index

- [Signal Denoising Documentation](docs/SIGNAL_DENOISING.md)
- [Exploratory Data Analysis Report](docs/EDA_REPORT.md)
- [Multi-Model & Ensemble Benchmark Report](docs/MODEL_BENCHMARK.md)
