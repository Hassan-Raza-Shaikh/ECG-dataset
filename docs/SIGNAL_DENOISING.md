# 🌊 Electrocardiogram (ECG) Signal Processing & Denoising

Clinical 12-lead Electrocardiography (ECG) recordings are often contaminated with various types of noise artifacts during acquisition. Effective signal preprocessing and denoising are critical steps to ensure high diagnostic fidelity and optimal Machine Learning model performance.

---

## 🎛️ ECG Noise Artifacts & Denoising Architecture

Our `ECGPreprocessor` module (`src/preprocessing.py`) implements a multi-stage zero-phase digital filtering and wave-decomposition pipeline:

```
Raw 12-Lead ECG Signal (100 Hz / 500 Hz)
    │
    ├──► 1. High-Pass Butterworth Filter (0.5 Hz Cutoff) ──► Baseline Wander Removal
    │
    ├──► 2. Low-Pass Butterworth Filter (40.0 Hz Cutoff) ──► Muscle (EMG) Noise Removal
    │
    ├──► 3. IIR Notch Filter (50.0 Hz / 60.0 Hz) ──────────► Powerline Hum Elimination
    │
    ├──► 4. Discrete Wavelet Transform (DWT sym8) ─────────► Transient Noise Soft-Thresholding
    │
    └──► 5. Per-Lead Z-Score Normalization ───────────────► Zero-Mean Unit-Variance Scaling
```

---

## 🔬 Mathematical & Filtering Details

### 1. Baseline Wander Removal (High-Pass Filter)
- **Artifact Source**: Caused by patient respiration, body movement, and electrode impedance changes (< 0.5 Hz).
- **Filter Specifications**: 5th-order zero-phase Butterworth high-pass filter with a **0.5 Hz cutoff frequency**.
- **Implementation**: Utilizes `scipy.signal.filtfilt` to achieve zero phase distortion, preserving temporal alignment of cardiac intervals (P-wave, QRS complex, T-wave).

### 2. Muscle Artifact (EMG) Noise Removal (Low-Pass Filter)
- **Artifact Source**: High-frequency electrical activity from skeletal muscle contraction (> 35 - 40 Hz).
- **Filter Specifications**: 5th-order zero-phase Butterworth low-pass filter with a **40.0 Hz cutoff frequency**.
- **Implementation**: Suppresses high-frequency EMG noise while retaining essential QRS complex morphology (0.5–40 Hz clinical ECG bandwidth).

### 3. Powerline Interference Removal (Notch Filter)
- **Artifact Source**: AC electrical grid hum operating at 50 Hz (Europe/Asia) or 60 Hz (Americas).
- **Filter Specifications**: Infinite Impulse Response (IIR) Notch filter centered at **50.0 Hz** with quality factor $Q = 30.0$.
- **Implementation**: Utilizes `scipy.signal.iirnotch` to eliminate narrow-band powerline hum.

### 4. Wavelet Denoising (Discrete Wavelet Transform - DWT)
- **Artifact Source**: Non-stationary random noise spikes and background white noise.
- **Wavelet Selection**: Symlet wavelet `sym8` with 4-level decomposition.
- **Thresholding Strategy**: Donoho-Johnstone Universal Threshold $\lambda$:
  $$\lambda = \sigma \sqrt{2 \ln(N)}$$
  where $\sigma$ is estimated from the median absolute deviation (MAD) of the finest scale detail coefficients ($d_1$):
  $$\sigma = \frac{\text{median}(|d_1|)}{0.6745}$$
- **Soft Thresholding**: Applied to detail coefficients while retaining approximation coefficients intact:
  $$\eta_{\text{soft}}(x, \lambda) = \text{sign}(x) \cdot \max(|x| - \lambda, 0)$$

### 5. Per-Lead Z-Score Normalization
- **Purpose**: Normalizes amplitude scales across different recording devices and leads.
- **Equation**:
  $$\hat{X}_i = \frac{X_i - \mu_i}{\sigma_i + \epsilon}$$
  where $\mu_i$ and $\sigma_i$ are the mean and standard deviation of lead $i$.

---

## 💻 Python Usage Example

```python
from src.preprocessing import ECGPreprocessor
import numpy as np

# Instantiate Preprocessor
preprocessor = ECGPreprocessor(
    sampling_rate=100,
    highpass_cutoff=0.5,
    lowpass_cutoff=40.0,
    notch_freq=50.0,
    apply_wavelet=True,
    normalize="zscore"
)

# Raw signal shape: (1000, 12)
raw_ecg = np.random.randn(1000, 12)
clean_ecg = preprocessor.process(raw_ecg)
```
