"""
================================================================================
STEP 5: TRAIN DEEP LEARNING RESNET1D MODEL (LAYMAN SCRIPT)
================================================================================
Trains the 1D Deep Residual ConvNet (ResNet1D) architecture.
Usage:
  PYTHONPATH=. python3 src/05_train_deep_learning_resnet.py
================================================================================
"""

from src.train import train_pipeline

if __name__ == "__main__":
    train_pipeline()
