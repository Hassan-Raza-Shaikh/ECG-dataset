"""
================================================================================
STEP 5c: 1D ViT (OVERLAPPING PATCHES) & 1D AUGMENTATION (500 Hz) (LAYMAN SCRIPT)
================================================================================
Trains a 1D Vision Transformer on 500 Hz ECG signals using only 4 selected leads
with 1D data augmentation and overlapping temporal patch tokenization.
Usage:
  PYTHONPATH=. python3 src/05c_train_vit_500hz_augmented.py
================================================================================
"""

from src.train_vit_500hz_augmented import train_vit_500hz_pipeline

if __name__ == "__main__":
    train_vit_500hz_pipeline(epochs=5, batch_size=64, lr=1e-3)
