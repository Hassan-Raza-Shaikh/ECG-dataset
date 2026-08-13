# 🫀 PTB-XL 12-Lead ECG Machine Learning & Signal Processing Pipeline

A production-grade, state-of-the-art end-to-end Machine Learning and Signal Denoising pipeline built for the **PTB-XL 12-lead Electrocardiography Dataset** (21,837 clinical 10-second ECG records from 18,885 patients).

---

## 📁 Repository Structure

```
ECG-dataset/
├── data/
│   └── raw/
│       └── ptbxl/             # Cleaned PTB-XL dataset (ptbxl_database.csv, records100, scp_statements.csv)
├── src/
│   ├── preprocessing.py       # ECG Denoising: Butterworth Highpass/Lowpass, Notch filter, DWT Wavelet Denoising & Normalization
│   ├── dataset.py             # PyTorch Dataset & DataLoaders with official PTB-XL benchmark fold splits
│   ├── eda.py                 # Exploratory Data Analysis & 12-lead signal visualization
│   ├── models.py              # ResNet1D & CNN-BiLSTM-Attention deep neural network architectures
│   ├── train.py               # End-to-end training pipeline with loss & AUC tracking
│   └── evaluate.py            # Test set evaluation (Fold 10) & ROC Curve generator
├── configs/
│   └── config.yaml            # Hyperparameters, filter cutoffs, and pipeline paths
├── artifacts/
│   ├── eda/                   # Saved EDA plots (demographics, class distribution, raw vs denoised ECG)
│   └── models/                # Saved best model checkpoints, ROC curves, and training metrics
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation & usage guide
```

---

## 🎛️ Signal Processing & Denoising Pipeline

Clinical ECG signals suffer from noise artifacts. Our `ECGPreprocessor` applies a multi-stage filtering pipeline:

1. **Baseline Wander Removal (High-Pass Filter)**: 5th-order zero-phase Butterworth filter with a 0.5 Hz cutoff frequency to eliminate low-frequency breathing/movement artifacts.
2. **EMG Muscle Noise Removal (Low-Pass Filter)**: 5th-order Butterworth filter with a 40 Hz cutoff frequency to remove high-frequency muscular contraction noise.
3. **Powerline Interference Removal (Notch Filter)**: IIR Notch filter targeting 50 Hz (or 60 Hz) AC electrical hum.
4. **Wavelet Denoising (DWT)**: Discrete Wavelet Transform using `sym8` wavelets with Donoho-Johnstone soft-thresholding to suppress high-frequency noise while preserving sharp QRS complex boundaries.
5. **Per-Lead Normalization**: Zero-mean, unit-variance Z-score normalization per lead.

---

## 🏗️ Model Architectures

The pipeline includes two SOTA models tailored for 12-lead continuous 1D signals:

- **ResNet1D (Deep 1D Residual Convolutional Network)**:
  - 12 input channels × 1000 temporal samples (10s @ 100 Hz).
  - 4 residual stages with 1D convolutions, batch normalization, ReLU, and shortcut skip connections.
  - Global Adaptive Average Pooling + Classification Head targeting 5 SCP Superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
- **CNN-BiLSTM-Attention**:
  - 1D-CNN feature extractor for local morphological patterns (P-wave, QRS, T-wave).
  - Bidirectional LSTM layers capturing temporal sequence dynamics across cardiac cycles.
  - Self-Attention mechanism weighting important temporal intervals.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
Generates demographic summaries, diagnostic class distributions, co-occurrence heatmaps, and raw vs. denoised 12-lead ECG waveform comparisons:
```bash
PYTHONPATH=. python3 src/eda.py
```
Outputs saved in `artifacts/eda/`.

### 3. Train the Model
Trains the model using official benchmark folds (Folds 1-8 for training, Fold 9 for validation):
```bash
PYTHONPATH=. python3 src/train.py
```
Checkpoints and training curves saved in `artifacts/models/`.

### 4. Evaluate on Held-Out Test Set (Fold 10)
Evaluates the best saved checkpoint on Fold 10 and plots multi-class ROC curves:
```bash
PYTHONPATH=. python3 src/evaluate.py
```
Outputs saved in `artifacts/models/test_roc_curves.png` and `test_metrics_summary.csv`.

---

## 📊 Diagnostic Classes (SCP Superclasses)

1. `NORM`: Normal ECG
2. `MI`: Myocardial Infarction
3. `STTC`: ST/T Change
4. `CD`: Conduction Disturbance
5. `HYP`: Hypertrophy
