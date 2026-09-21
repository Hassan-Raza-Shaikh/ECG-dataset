"""
================================================================================
🫀 STEP 5b: VISION TRANSFORMER (ViT) ON SELECTED LEADS
================================================================================
Trains and evaluates a 1D Vision Transformer (ECG_ViT1D) using ONLY the 
top selected unique channels (e.g. ['aVF', 'III', 'I', 'II']) identified in Step 3.

Pipeline:
  1. Load Preprocessed 12-lead ECG signals
  2. Extract only the 4 selected channels: [aVF, III, I, II]
  3. Train 1D Vision Transformer (Patch Embeddings -> MHSA Transformer -> [CLS] Token Head)
  4. Evaluate on held-out Test Set (Fold 10)
  5. Save metrics and plots (ROC curves, training history)
================================================================================
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import compute_superclasses, SUPERCLASSES
from src.models import build_model
from src.preprocessing import ECGPreprocessor

SELECTED_LEADS = ["aVF", "III", "I", "II"]
LEAD_INDICES = [5, 2, 0, 1]


class SelectedLeadsDataset(Dataset):
    def __init__(self, signals, labels, lead_indices=LEAD_INDICES):
        # signals shape: (N, time_steps, 12) -> select leads -> transpose to (N, num_selected_leads, time_steps)
        self.signals = signals[:, :, lead_indices]
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx] # (time_steps, 4)
        sig_tensor = torch.tensor(sig.T, dtype=torch.float32) # (4, time_steps)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sig_tensor, label_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(dataloader, desc="Training ViT", leave=False):
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
    all_preds = []
    all_targets = []

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

    # Compute Macro ROC-AUC
    aucs = []
    for c in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, c])) > 1:
            aucs.append(roc_auc_score(all_targets[:, c], all_preds[:, c]))
    macro_auc = np.mean(aucs) if aucs else 0.5

    return epoch_loss, macro_auc, all_preds, all_targets


def train_vit_pipeline(config_path="configs/config.yaml", epochs=5, batch_size=64, lr=1e-3):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    db_path = config["data"]["database_csv"]
    scp_path = config["data"]["scp_csv"]
    sampling_rate = config["data"]["sampling_rate"]

    print("=" * 75)
    print("STEP 5b: VISION TRANSFORMER (ViT) TRAINING ON SELECTED LEADS")
    print(f"Selected Channels: {SELECTED_LEADS} (Lead Indices: {LEAD_INDICES})")
    print("=" * 75)

    # 1. Load Metadata
    print(f"Loading metadata from {db_path}...")
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)
    df = compute_superclasses(df, scp_df)

    # 2. Load Signal Cache
    cache_path = os.path.join(raw_dir, f"signals_cache_{sampling_rate}hz.npy")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(raw_dir, "signals_cache_100hz.npy")
        sampling_rate = 100

    print(f"Loading signals from {cache_path}...")
    all_signals = np.load(cache_path) # (N, time_steps, 12)
    time_steps = all_signals.shape[1]

    # Preprocess signals if not pre-cached
    preprocessor = ECGPreprocessor(sampling_rate=sampling_rate, apply_wavelet=True, normalize="zscore")
    print("Applying z-score and wavelet filtering...")
    # Normalize per lead across all signals
    means = np.mean(all_signals, axis=1, keepdims=True)
    stds = np.std(all_signals, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    all_signals = (all_signals - means) / stds

    # 3. Train/Val/Test Split
    train_folds = config["split"]["train_folds"]
    val_folds = config["split"]["val_folds"]
    test_folds = config["split"]["test_folds"]

    train_idx = df.strat_fold.isin(train_folds).values
    val_idx = df.strat_fold.isin(val_folds).values
    test_idx = df.strat_fold.isin(test_folds).values

    X_train, y_train = all_signals[train_idx], df[SUPERCLASSES].values[train_idx]
    X_val, y_val = all_signals[val_idx], df[SUPERCLASSES].values[val_idx]
    X_test, y_test = all_signals[test_idx], df[SUPERCLASSES].values[test_idx]

    print(f"Data Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    train_ds = SelectedLeadsDataset(X_train, y_train, LEAD_INDICES)
    val_ds = SelectedLeadsDataset(X_val, y_val, LEAD_INDICES)
    test_ds = SelectedLeadsDataset(X_test, y_test, LEAD_INDICES)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Initialize 1D Vision Transformer
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {device}")

    patch_size = 50 if time_steps <= 1000 else 100
    model = build_model(
        "ECG_ViT1D",
        in_channels=len(SELECTED_LEADS),
        num_classes=len(SUPERCLASSES),
        seq_len=time_steps
    ).to(device)

    print(f"Initialized ECG_ViT1D: in_channels={len(SELECTED_LEADS)}, seq_len={time_steps}, patch_size={patch_size}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    best_val_auc = 0.0
    os.makedirs("artifacts/models", exist_ok=True)
    best_ckpt_path = "artifacts/models/best_vit_selected_leads.pth"

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print("\nStarting ViT Training on Selected Leads...")
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

    # 6. Evaluation on Held-Out Test Set (Fold 10)
    print("\nLoading best checkpoint for official Test Set (Fold 10) evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_auc, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)

    # Binary predictions with threshold 0.5
    test_pred_bin = (test_preds >= 0.5).astype(int)
    test_f1 = f1_score(test_targets, test_pred_bin, average="macro", zero_division=0)
    test_prec = precision_score(test_targets, test_pred_bin, average="macro", zero_division=0)
    test_rec = recall_score(test_targets, test_pred_bin, average="macro", zero_division=0)

    print("\n" + "=" * 75)
    print("  ViT (SELECTED LEADS) FINAL TEST SET RESULTS (FOLD 10)")
    print("=" * 75)
    print(f"Test Macro ROC-AUC : {test_auc:.4f} ({test_auc*100:.2f}%)")
    print(f"Test Macro F1      : {test_f1:.4f}")
    print(f"Test Precision     : {test_prec:.4f}")
    print(f"Test Recall        : {test_rec:.4f}")
    print("=" * 75)

    # Per-Class Metrics Table
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
    print("\nPer-Class Breakdown on Selected Leads (ViT):")
    print(class_df.to_string(index=False))

    # Save to docs/results/
    os.makedirs("docs/results", exist_ok=True)
    res_path = "docs/results/vit_selected_leads_metrics.csv"
    class_df.to_csv(res_path, index=False)
    print(f"\nSaved test metrics to {res_path}")

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
    plt.title(f"Vision Transformer (ViT) on Selected Leads {SELECTED_LEADS}\nTest Set Macro ROC-AUC: {test_auc:.4f}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    roc_plot_path = "artifacts/models/vit_selected_leads_roc_curves.png"
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
    plt.title("ViT Loss History")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["val_auc"], "g-o", label="Val Macro AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Macro ROC-AUC")
    plt.title("ViT Validation AUC History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    hist_plot_path = "artifacts/models/vit_training_history.png"
    plt.savefig(hist_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training history to {hist_plot_path}")

    return test_auc, class_df


if __name__ == "__main__":
    train_vit_pipeline(epochs=5, batch_size=64, lr=1e-3)
