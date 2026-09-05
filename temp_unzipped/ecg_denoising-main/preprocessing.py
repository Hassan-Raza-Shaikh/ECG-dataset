import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import pywt


class ECGPreprocessor:
    """
    Comprehensive ECG Signal Preprocessor and Denoising Pipeline.
    
    Supports:
    - High-pass Butterworth filter (removes baseline wander ~0.5 Hz)
    - Low-pass Butterworth filter (removes muscle EMG artifacts >40 Hz)
    - Notch filter (removes 50/60 Hz powerline hum)
    - Wavelet Denoising (DWT soft-thresholding with sym8/db4 wavelets)
    - Per-lead Z-Score Normalization
    """

    def __init__(
        self,
        sampling_rate=500,
        highpass_cutoff=0.5,
        lowpass_cutoff=40.0,
        notch_freq=50.0,
        notch_q=30.0,
        apply_wavelet=True,
        wavelet_name="sym8",
        wavelet_level=5,
        normalize="zscore",
    ):
        self.sampling_rate = sampling_rate
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.apply_wavelet = apply_wavelet
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.normalize_method = normalize

    def highpass_filter(self, signal, order=5):
        """Applies zero-phase high-pass Butterworth filter to remove baseline wander."""
        nyquist = 0.5 * self.sampling_rate
        cutoff = self.highpass_cutoff / nyquist
        b, a = butter(order, cutoff, btype="highpass")
        return filtfilt(b, a, signal, axis=0)

    def lowpass_filter(self, signal, order=5):
        """Applies zero-phase low-pass Butterworth filter to remove EMG muscle artifacts."""
        nyquist = 0.5 * self.sampling_rate
        cutoff = min(self.lowpass_cutoff / nyquist, 0.99)
        b, a = butter(order, cutoff, btype="lowpass")
        return filtfilt(b, a, signal, axis=0)

    def notch_filter(self, signal):
        """Applies IIR Notch filter to eliminate powerline interference."""
        nyquist = 0.5 * self.sampling_rate
        if self.notch_freq >= nyquist:
            return signal # Notch freq exceeds Nyquist rate (e.g. 50Hz notch on 100Hz sampling)
        b, a = iirnotch(self.notch_freq, self.notch_q, fs=self.sampling_rate)
        return filtfilt(b, a, signal, axis=0)

    def wavelet_denoise_lead(self, lead_signal):
        """Denoises a 1D single-lead signal using Discrete Wavelet Transform (DWT) soft thresholding."""
        try:
            coeffs = pywt.wavedec(lead_signal, self.wavelet_name, level=self.wavelet_level)
            sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
            if sigma > 1e-8:
                uthresh = sigma * np.sqrt(2 * np.log(len(lead_signal)))
                new_coeffs = [coeffs[0]] + [
                    pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]
                ]
                denoised = pywt.waverec(new_coeffs, self.wavelet_name)
                return denoised[: len(lead_signal)].astype(np.float32)
            return lead_signal
        except Exception:
            return lead_signal

    def wavelet_denoise(self, signal):
        """Denoises multi-lead ECG signal (time x leads or leads x time)."""
        # Assume input shape is (time_steps, num_leads)
        denoised = np.zeros_like(signal)
        for lead_idx in range(signal.shape[1]):
            denoised[:, lead_idx] = self.wavelet_denoise_lead(signal[:, lead_idx])
        return denoised

    def normalize(self, signal):
        """Normalizes signal per lead."""
        if self.normalize_method == "zscore":
            mean = np.mean(signal, axis=0, keepdims=True)
            std = np.std(signal, axis=0, keepdims=True)
            std[std == 0] = 1.0 # Prevent division by zero
            return (signal - mean) / std
        elif self.normalize_method == "minmax":
            min_val = np.min(signal, axis=0, keepdims=True)
            max_val = np.max(signal, axis=0, keepdims=True)
            diff = max_val - min_val
            diff[diff == 0] = 1.0
            return (signal - min_val) / diff
        return signal

    def process(self, signal):
        """
        Executes the complete preprocessing and denoising pipeline on raw 12-lead ECG signal.
        
        Input: numpy array of shape (time_steps, num_leads) e.g., (1000, 12)
        Output: numpy array of shape (time_steps, num_leads)
        """
        processed = signal.copy().astype(np.float32)

        # 1. Remove baseline wander (High-pass filter)
        if self.highpass_cutoff > 0:
            processed = self.highpass_filter(processed)

        # 2. Remove high-frequency noise (Low-pass filter)
        if self.lowpass_cutoff > 0:
            processed = self.lowpass_filter(processed)

        # 3. Remove powerline hum (Notch filter)
        if self.notch_freq > 0:
            processed = self.notch_filter(processed)

        # 4. Wavelet soft-thresholding
        if self.apply_wavelet:
            processed = self.wavelet_denoise(processed)

        # 5. Per-lead Normalization
        if self.normalize_method:
            processed = self.normalize(processed)

        return processed