# 🫀 PTB-XL 12-Lead ECG Machine Learning & Signal Processing Pipeline

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Dataset](https://img.shields.io/badge/Dataset-PTB--XL--v1.0.3-green.svg)
![Open Source](https://img.shields.io/badge/Open_Source-Public-brightgreen.svg)

A production-grade, state-of-the-art end-to-end Machine Learning, Signal Denoising, and Multi-Model Benchmarking pipeline built for the **PTB-XL 12-lead Electrocardiography Dataset** (21,837 clinical 10-second ECG records from 18,885 patients).

---

## 🔄 100% Replication & Reproducibility Guide

Anyone with this repository can replicate the exact benchmark results, signal denoising, EDA, and model training in **3 simple steps**:

### 1️⃣ Clone Repository & Install Dependencies
```bash
git clone https://github.com/Hassan-Raza-Shaikh/ECG-dataset.git
cd ECG-dataset
pip install -r requirements.txt
```

### 2️⃣ Download PTB-XL Dataset
You can automatically download and extract the dataset (v1.0.3) from PhysioNet into `data/raw/ptbxl/`:
```bash
python src/download_dataset.py
```
*(Or manually place `ptbxl_database.csv`, `scp_statements.csv`, `records100/`, `records500/` inside `data/raw/ptbxl/`)*.

### 3️⃣ Run End-to-End Pipeline & Benchmark
Run the single master command to execute signal denoising, automated EDA, feature extraction, model training, and stacking ensemble evaluation:
```bash
PYTHONPATH=. python3 src/benchmark.py
```

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
├── data/
│   └── raw/
│       └── ptbxl/             # Cleaned PTB-XL dataset & binary cache files
├── src/
│   ├── download_dataset.py    # Auto-downloader for PTB-XL dataset from PhysioNet
│   ├── preprocessing.py       # ECG Denoising: Butterworth Highpass/Lowpass, Notch filter, DWT Wavelet Denoising & Normalization
│   ├── dataset.py             # PyTorch Dataset & DataLoaders with official PTB-XL benchmark fold splits
│   ├── features.py            # Extract 192 time-domain, spectral PSD, and wavelet energy features per ECG
│   ├── eda.py                 # Exploratory Data Analysis & 12-lead signal visualization
│   ├── models.py              # ResNet1D & CNN-BiLSTM-Attention deep neural network architectures
│   ├── train.py               # ResNet1D training pipeline with PyTorch, AdamW & Cosine Scheduler
│   ├── train_traditional.py   # Trainer for RandomForest, ExtraTrees, HistGradientBoosting, Linear SVM, LogisticRegression
│   ├── benchmark.py           # Multi-model benchmark suite & Weighted Stacking Ensemble generator
│   ├── build_cache.py         # Multi-threaded binary dataset cache builder (1,700 records/sec)
│   └── evaluate.py            # Test set evaluation (Fold 10) & ROC Curve generator
├── docs/
│   ├── SIGNAL_DENOISING.md    # Technical guide on ECG signal preprocessing & filtering math
│   ├── EDA_REPORT.md          # Comprehensive Exploratory Data Analysis report
│   └── MODEL_BENCHMARK.md     # Detailed Multi-Model & Stacking Ensemble benchmark leaderboard
├── configs/
│   └── config.yaml            # Hyperparameters, filter cutoffs, and pipeline paths
├── artifacts/
│   ├── eda/                   # Saved EDA plots (demographics, class distribution, raw vs denoised ECG)
│   └── models/                # Saved best model checkpoints, ROC curves, and training metrics
├── .gitignore                 # Git ignore rules for dataset binaries & model checkpoints
├── requirements.txt           # Python package dependencies
└── README.md                  # Master project guide & documentation
```

---

## 📚 Technical Documentation Index

- [Signal Denoising Documentation](docs/SIGNAL_DENOISING.md)
- [Exploratory Data Analysis Report](docs/EDA_REPORT.md)
- [Multi-Model & Ensemble Benchmark Report](docs/MODEL_BENCHMARK.md)
