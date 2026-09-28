"""
================================================================================
🫀 STEP 5c: 1D VISION TRANSFORMER (OVERLAPPING PATCHES) & 1D AUGMENTATION (500 Hz)
================================================================================
Trains and evaluates a 1D Vision Transformer on high-resolution 500 Hz ECG signals 
(5,000 samples per lead) for the 4 selected unique leads [aVF, III, I, II] with:
  1. 1D Signal Data Augmentation (Amplitude scaling, Time jitter, Gaussian noise, Cutout)
  2. Overlapping Patch Tokenization (Patch size = 100, Stride = 50 -> 50% overlap, 99 tokens)
  3. Pre-LN Multi-Head Self-Attention Transformer Encoder
  4. Full Evaluation on Held-Out Test Set (Fold 10)
================================================================================
"""

import os
import yaml
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
import wfdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import compute_superclasses, SUPERCLASSES
from src.models import build_model
from src.augmentation import ECGAugmenter1D

SELECTED_LEADS = ["aVF", "III", "I", "II"]
LEAD_INDICES = [5, 2, 0, 1] # Indices in PTB-XL 12-lead array


def _read_single_500hz_selected(fpath):
    """Worker function to read 500 Hz signal and keep only 4 selected leads."""
    sig, _ = wfdb.rdsamp(fpath) # (5000, 12)
    selected = sig[:, LEAD_INDICES].astype(np.float32) # (5000, 4)
    # Per-lead z-score normalization
    mean = np.mean(selected, axis=0, keepdims=True)
    std = np.std(selected, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((selected - mean) / std).astype(np.float32)


def load_or_create_500hz_cache(df, raw_dir):
    """Loads or generates binary cache of the 4 selected leads at 500 Hz."""
    cache_path = os.path.join(raw_dir, "signals_cache_500hz_selected.npy")
    if os.path.exists(cache_path):
        print(f"Loading cached 500 Hz selected signals from {cache_path}...")
        return np.load(cache_path)

    filepaths = [os.path.join(raw_dir, f) for f in df["filename_hr"]]
    print(f"Reading 500 Hz signals across CPU cores ({len(filepaths)} records)...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        signals = list(executor.map(_read_single_500hz_selected, filepaths, chunksize=100))

    signals = np.array(signals, dtype=np.float32) # Shape: (21799, 5000, 4)
    print(f"Saving 500 Hz selected signals cache to {cache_path} (Shape: {signals.shape})...")
    np.save(cache_path, signals)
    return signals


class AugmentedSelectedLeadsDataset500Hz(Dataset):
    """Dataset with on-the-fly 1D data augmentation for 500 Hz signals."""

    def __init__(self, signals, labels, augmenter=None):
        # signals shape: (N, 5000, 4)
        self.signals = signals
        self.labels = labels
        self.augmenter = augmenter

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx] # (5000, 4)
        sig_tensor = torch.tensor(sig.T, dtype=torch.float32) # (4, 5000)

        if self.augmenter is not None:
            sig_tensor = self.augmenter(sig_tensor)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sig_tensor, label_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(dataloader, desc="Training 500Hz ViT", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(y)
    return running_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * len(y)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    epoch_loss = running_loss / len(dataloader.dataset)

    aucs = []
    for c in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, c])) > 1:
            aucs.append(roc_auc_score(all_targets[:, c], all_preds[:, c]))
    macro_auc = np.mean(aucs) if aucs else 0.5

    return epoch_loss, macro_auc, all_preds, all_targets


def train_vit_500hz_pipeline(config_path="configs/config.yaml", epochs=5, batch_size=64, lr=1e-3):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    db_path = config["data"]["database_csv"]
    scp_path = config["data"]["scp_csv"]

    print("=" * 80)
    print("STEP 5c: 1D ViT (OVERLAPPING PATCHES) & 1D AUGMENTATION ON 500 Hz DATASET")
    print(f"Selected Channels: {SELECTED_LEADS} (Indices: {LEAD_INDICES})")
    print("Patch Config: size = 100 samples (0.2s), stride = 50 samples (50% overlap -> 99 tokens)")
    print("=" * 80)

    # 1. Load Metadata & 500 Hz Signals
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)
    df = compute_superclasses(df, scp_df)

    signals_500hz = load_or_create_500hz_cache(df, raw_dir)
    print(f"Loaded 500 Hz signals shape: {signals_500hz.shape}")

    # 2. Train / Val / Test Split
    train_folds = config["split"]["train_folds"]
    val_folds = config["split"]["val_folds"]
    test_folds = config["split"]["test_folds"]

    train_idx = df.strat_fold.isin(train_folds).values
    val_idx = df.strat_fold.isin(val_folds).values
    test_idx = df.strat_fold.isin(test_folds).values

    X_train, y_train = signals_500hz[train_idx], df[SUPERCLASSES].values[train_idx]
    X_val, y_val = signals_500hz[val_idx], df[SUPERCLASSES].values[val_idx]
    X_test, y_test = signals_500hz[test_idx], df[SUPERCLASSES].values[test_idx]

    print(f"Data Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 3. Create Datasets with 1D Augmenter for Training
    augmenter = ECGAugmenter1D(
        min_scale=0.85,
        max_scale=1.15,
        max_shift=100,
        noise_std=0.02,
        cutout_holes=2,
        cutout_max_len=100,
        p=0.5
    )
    train_ds = AugmentedSelectedLeadsDataset500Hz(X_train, y_train, augmenter=augmenter)
    val_ds = AugmentedSelectedLeadsDataset500Hz(X_val, y_val, augmenter=None)
    test_ds = AugmentedSelectedLeadsDataset500Hz(X_test, y_test, augmenter=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Initialize Overlapping Patch 1D Vision Transformer
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {device}")

    model = build_model(
        "ECG_ViT1D",
        in_channels=len(SELECTED_LEADS),
        num_classes=len(SUPERCLASSES),
        seq_len=5000,
        patch_size=100,
        patch_stride=50
    ).to(device)

    print(f"Initialized Overlapping ECG_ViT1D: in_channels=4, seq_len=5000, patches={model.num_patches}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    best_val_auc = 0.0
    os.makedirs("artifacts/models", exist_ok=True)
    best_ckpt_path = "artifacts/models/best_vit_500hz_overlapping.pth"
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print("\nStarting 500 Hz Overlapping ViT Training...")
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_auc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_auc"].append(v_auc)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Macro AUC: {v_auc:.4f}")

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  >>> Checkpoint saved! Best Val AUC: {best_val_auc:.4f}")

    # 6. Evaluation on Official Held-Out Test Set (Fold 10)
    print("\nEvaluating best checkpoint on Test Set (Fold 10)...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_auc, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)

    test_pred_bin = (test_preds >= 0.5).astype(int)
    test_f1 = f1_score(test_targets, test_pred_bin, average="macro", zero_division=0)
    test_prec = precision_score(test_targets, test_pred_bin, average="macro", zero_division=0)
    test_rec = recall_score(test_targets, test_pred_bin, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print("  500 Hz OVERLAPPING ViT (SELECTED LEADS) FINAL TEST RESULTS (FOLD 10)")
    print("=" * 80)
    print(f"Test Macro ROC-AUC : {test_auc:.4f} ({test_auc*100:.2f}%)")
    print(f"Test Macro F1      : {test_f1:.4f}")
    print(f"Test Precision     : {test_prec:.4f}")
    print(f"Test Recall        : {test_rec:.4f}")
    print("=" * 80)

    # Per-Class Breakdown
    class_rows = []
    for i, c in enumerate(SUPERCLASSES):
        c_auc = roc_auc_score(test_targets[:, i], test_preds[:, i])
        c_f1 = f1_score(test_targets[:, i], test_pred_bin[:, i], zero_division=0)
        c_prec = precision_score(test_targets[:, i], test_pred_bin[:, i], zero_division=0)
        c_rec = recall_score(test_targets[:, i], test_pred_bin[:, i], zero_division=0)
        class_rows.append({
            "Superclass": c,
            "ROC_AUC": c_auc,
            "F1_Score": c_f1,
            "Precision": c_prec,
            "Recall": c_rec,
            "Support": int(test_targets[:, i].sum())
        })

    class_df = pd.DataFrame(class_rows)
    print("\nPer-Class Breakdown on 500 Hz Overlapping ViT:")
    print(class_df.to_string(index=False))

    # Save to docs/results/
    os.makedirs("docs/results", exist_ok=True)
    res_path = "docs/results/vit_500hz_overlapping_metrics.csv"
    class_df.to_csv(res_path, index=False)
    print(f"\nSaved metrics to {res_path}")

    # 7. Plot ROC Curves
    plt.figure(figsize=(9, 7))
    for i, c in enumerate(SUPERCLASSES):
        fpr, tpr, _ = roc_curve(test_targets[:, i], test_preds[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(f"500 Hz Overlapping ViT (Selected Leads {SELECTED_LEADS})\nTest Macro ROC-AUC: {test_auc:.4f}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    roc_plot_path = "artifacts/models/vit_500hz_overlapping_roc_curves.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves plot to {roc_plot_path}")

    # 8. Plot Training History
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], "b-o", label="Train Loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], "r-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("500 Hz ViT Loss History")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["val_auc"], "g-o", label="Val Macro AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Macro ROC-AUC")
    plt.title("500 Hz ViT Validation AUC History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    hist_plot_path = "artifacts/models/vit_500hz_overlapping_training_history.png"
    plt.savefig(hist_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training history to {hist_plot_path}")

    return test_auc, class_df


if __name__ == "__main__":
    train_vit_500hz_pipeline(epochs=5, batch_size=64, lr=1e-3)
