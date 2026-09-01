"""
================================================================================
STEP 4: TRAIN CLASSICAL ML & TREE ENSEMBLE MODELS (LAYMAN SCRIPT)
================================================================================
Extracts 192 features and trains Random Forest, HistGradientBoosting, ExtraTrees,
Linear SVM, and Logistic Regression.
Usage:
  PYTHONPATH=. python3 src/04_train_classical_ml_models.py
================================================================================
"""

from src.train_traditional import train_evaluate_traditional_models

if __name__ == "__main__":
    train_evaluate_traditional_models()
