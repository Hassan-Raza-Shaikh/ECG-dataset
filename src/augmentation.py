"""
================================================================================
🌊 1D ECG SIGNAL DATA AUGMENTATION MODULE
================================================================================
Implements domain-specific 1D temporal transforms for multi-lead ECG signals:
  1. RandomAmplitudeScale: Simulates chest impedance and electrode contact variations.
  2. RandomTimeShift: Circular or zero-padded temporal shift (phase jitter).
  3. RandomGaussianNoise: Low-amplitude background electrical noise injection.
  4. RandomCutout: Random temporal masking (simulates brief electrode detachment).
  5. ECGAugmenter1D: Composable augmentation pipeline for PyTorch training batches.
================================================================================
"""

import numpy as np
import torch


class RandomAmplitudeScale:
    """Scales signal amplitude by random factor alpha in [min_scale, max_scale]."""

    def __init__(self, min_scale=0.85, max_scale=1.15, p=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, x):
        # x: torch.Tensor of shape (channels, seq_len)
        if torch.rand(1).item() < self.p:
            scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
            return x * scale
        return x


class RandomTimeShift:
    """Shifts the signal temporally by up to max_shift samples with circular wrapping."""

    def __init__(self, max_shift=100, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x):
        # x: torch.Tensor of shape (channels, seq_len)
        if torch.rand(1).item() < self.p:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            return torch.roll(x, shifts=shift, dims=-1)
        return x


class RandomGaussianNoise:
    """Injects low-amplitude Gaussian white noise."""

    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        # x: torch.Tensor of shape (channels, seq_len)
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class RandomCutout:
    """Randomly masks a brief temporal segment with zeros."""

    def __init__(self, num_holes=2, max_length=100, p=0.5):
        self.num_holes = num_holes
        self.max_length = max_length
        self.p = p

    def __call__(self, x):
        # x: torch.Tensor of shape (channels, seq_len)
        if torch.rand(1).item() < self.p:
            x_aug = x.clone()
            seq_len = x.shape[-1]
            for _ in range(self.num_holes):
                hole_len = torch.randint(10, self.max_length + 1, (1,)).item()
                start = torch.randint(0, max(1, seq_len - hole_len), (1,)).item()
                x_aug[:, start : start + hole_len] = 0.0
            return x_aug
        return x


class ECGAugmenter1D:
    """
    Composable 1D ECG Data Augmenter applying transforms sequentially.
    """

    def __init__(
        self,
        min_scale=0.85,
        max_scale=1.15,
        max_shift=100,
        noise_std=0.02,
        cutout_holes=2,
        cutout_max_len=100,
        p=0.5
    ):
        self.transforms = [
            RandomAmplitudeScale(min_scale, max_scale, p=p),
            RandomTimeShift(max_shift, p=p),
            RandomGaussianNoise(noise_std, p=p),
            RandomCutout(cutout_holes, cutout_max_len, p=p)
        ]

    def __call__(self, x):
        # x: torch.Tensor of shape (channels, seq_len)
        for t in self.transforms:
            x = t(x)
        return x
