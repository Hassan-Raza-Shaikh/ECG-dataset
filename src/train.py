import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

from src.dataset import get_dataloaders
from src.models import build_model


def get_device(requested_device):
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif (requested_device == "mps" or requested_device == "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif requested_device == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for signals, targets in dataloader:
            signals = signals.to(device)
            targets = targets.to(device)

            logits = model(signals)
            loss = criterion(logits, targets)

            total_loss += loss.item() * signals.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Compute Macro ROC-AUC
    try:
        macro_auc = roc_auc_score(all_targets, all_preds, average="macro")
    except Exception:
        macro_auc = 0.5

    # Compute per-class ROC-AUC
    per_class_auc = {}
    for i in range(all_targets.shape[1]):
        try:
            per_class_auc[i] = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except Exception:
            per_class_auc[i] = 0.5

    # Compute Macro F1-score (threshold at 0.5)
    binary_preds = (all_preds >= 0.5).astype(int)
    macro_f1 = f1_score(all_targets, binary_preds, average="macro", zero_division=0)

    return avg_loss, macro_auc, macro_f1, per_class_auc


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("BCE Loss History", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Validation Macro AUC plot
    axes[1].plot(history["val_auc"], label="Val Macro AUC", color="#2ca02c", linewidth=2)
    axes[1].plot(history["val_f1"], label="Val Macro F1", color="#d62728", linewidth=2)
    axes[1].set_title("Validation Metrics (Macro AUC & F1)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training history curves to {save_path}")


def train_pipeline(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    print(f"Using compute device: {device}")

    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Load DataLoaders
    print("\n--- Initializing DataLoaders ---")
    train_loader, val_loader, test_loader, superclasses = get_dataloaders(config)

    # Build Model
    model_name = config["training"]["model_name"]
    print(f"\nBuilding model architecture: {model_name}...")
    model = build_model(model_name, in_channels=12, num_classes=len(superclasses))
    model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_model_path = os.path.join(save_dir, f"best_{model_name}.pth")

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    print(f"\n--- Starting Model Training ({epochs} Epochs) ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (signals, targets) in enumerate(train_loader):
            signals = signals.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * signals.size(0)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                batch_loss = loss.item()
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Batch [{batch_idx+1:03d}/{len(train_loader):03d}] - Loss: {batch_loss:.4f}", flush=True)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_auc, val_f1, per_class_auc = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro AUC: {val_auc:.4f} | "
            f"Val Macro F1: {val_f1:.4f}",
            flush=True
        )

        # Save checkpoint if best validation AUC is achieved
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "superclasses": superclasses,
                "config": config,
            }, best_model_path)
            print(f"   Saved new best model checkpoint to {best_model_path} (AUC: {val_auc:.4f})", flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes!")
    print(f"Best Validation Macro AUC: {best_val_auc:.4f}")

    # Plot history curves
    plot_training_history(history, os.path.join(save_dir, f"{model_name}_training_history.png"))


if __name__ == "__main__":
    train_pipeline()
