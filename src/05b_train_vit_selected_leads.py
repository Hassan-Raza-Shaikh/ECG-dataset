"""
================================================================================
STEP 5b: TRAIN VISION TRANSFORMER (ViT) ON SELECTED LEADS (LAYMAN SCRIPT)
================================================================================
Trains a 1D Vision Transformer using only the top 4 selected leads [aVF, III, I, II]
identified in Step 3.
Usage:
  PYTHONPATH=. python3 src/05b_train_vit_selected_leads.py
================================================================================
"""

from src.train_vit_selected_leads import train_vit_pipeline

if __name__ == "__main__":
    train_vit_pipeline(epochs=5, batch_size=64, lr=1e-3)
