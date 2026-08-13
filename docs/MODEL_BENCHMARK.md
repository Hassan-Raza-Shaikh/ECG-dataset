# 🏆 Multi-Model Benchmark & Ensemble Performance Report

This report presents an empirical performance benchmark comparing **8 diverse Model Architectures, Tree Ensembles, Linear Classifiers, Deep Neural Networks, and a Weighted Stacking Ensemble** trained on the **PTB-XL 12-lead ECG dataset**.

---

## 📊 Final Benchmark Leaderboard (Held-Out Test Set Fold 10)

All models were trained on Folds 1–8 (17,418 records), tuned on Validation Fold 9 (2,183 records), and evaluated on the official **Held-Out Test Set (Fold 10, 2,198 records)**:

| Rank | Model Architecture / Ensemble | Model Category | Test Macro ROC-AUC | Test Macro F1 | Test Precision | Test Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Weighted Stacking Ensemble** | **Hybrid Deep + Tree Stacking** | **0.9182 (91.82%)** | **0.6759** | **0.8030** | **0.6218** |
| 🥈 **2** | **ResNet1D** | Deep Residual 1D CNN | **0.9079 (90.79%)** | **0.6696** | 0.7663 | **0.6342** |
| 🥉 **3** | **CNN-BiLSTM-Attention** | Spatial-Temporal Deep Net | **0.9028 (90.28%)** | **0.6761** | 0.7591 | 0.6333 |
| 4 | **HistGradientBoosting** | Histogram Tree Ensemble | **0.8675 (86.75%)** | 0.6033 | 0.7474 | 0.5231 |
| 5 | **RandomForest** | Random Decision Forests | **0.8512 (85.12%)** | 0.5431 | 0.7621 | 0.4528 |
| 6 | **ExtraTrees** | Extremely Randomized Trees | **0.8274 (82.74%)** | 0.4577 | 0.7704 | 0.3694 |
| 7 | **SVM_Linear** | Calibrated Support Vector Machine | **0.8263 (82.63%)** | 0.5179 | 0.7310 | 0.4270 |
| 8 | **LogisticRegression** | Regularized Linear Model | **0.8240 (82.40%)** | 0.5381 | 0.7147 | 0.4521 |

---

## 🎯 Per-Class Performance Breakdown (Best Model: ResNet1D & Stacking Ensemble)

### ResNet1D Per-Class Performance on Test Set (Fold 10):

| Diagnostic Superclass | Test ROC-AUC | Test F1-Score | Precision | Recall | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NORM** (Normal ECG) | **0.9428** | **0.8590** | 0.8257 | 0.8951 | 963 |
| **STTC** (ST/T Change) | **0.9311** | **0.7252** | 0.7959 | 0.6660 | 521 |
| **CD** (Conduction Disturbance) | **0.9239** | **0.7490** | 0.7647 | 0.7339 | 496 |
| **MI** (Myocardial Infarction) | **0.9221** | **0.7205** | 0.7550 | 0.6891 | 550 |
| **HYP** (Hypertrophy) | **0.8198** | **0.2943** | 0.6901 | 0.1870 | 262 |
| **MACRO AVERAGE** | **0.9079** | **0.6696** | **0.7663** | **0.6342** | **2,198** |

---

## ⚙️ Feature Extraction Details for Traditional Models

To train classical ML models (Random Forest, HistGradientBoosting, ExtraTrees, SVM, LogisticRegression), `src/features.py` extracts **192 biological features** per 10-second 12-lead ECG recording:

1. **Time-Domain Features (84 features)**: Mean, Std, Variance, Skewness, Kurtosis, Peak-to-Peak amplitude, Root Mean Square (RMS) per lead.
2. **Frequency-Domain Features (60 features)**: Welch Power Spectral Density (PSD) energy across 5 bands (0.5–4 Hz, 4–8 Hz, 8–15 Hz, 15–30 Hz, 30–40 Hz).
3. **Wavelet Energy Features (48 features)**: Discrete Wavelet Transform (`sym8` level 4) detail coefficient energies.

---

## 🎨 Benchmark Plots

Saved under `artifacts/models/`:
- `artifacts/models/model_auc_comparison_bar.png`: Test ROC-AUC ranking bar chart across all models.
- `artifacts/models/multi_model_roc_curves.png`: Overlay ROC curves comparison plot.
- `artifacts/models/test_roc_curves.png`: Multi-class ROC curves for ResNet1D.
- `artifacts/models/ResNet1D_training_history.png`: Training/Validation Loss and AUC curves over 5 epochs.
