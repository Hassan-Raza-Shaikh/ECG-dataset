import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, classification_report
from src.dataset import get_dataloaders
from src.models import build_model
from src.train import get_device


def plot_roc_curves(all_targets, all_preds, superclasses, save_path):
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    for i, sclass in enumerate(superclasses):
        fpr, tpr, _ = roc_curve(all_targets[:, i], all_preds[:, i])
        auc_score = roc_auc_score(all_targets[:, i], all_preds[:, i])
        plt.plot(fpr, tpr, label=f"{sclass} (AUC = {auc_score:.4f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves - Test Set (Fold 10)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves plot to {save_path}")


def evaluate_test_set(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    print(f"Using compute device for evaluation: {device}")

    save_dir = config["training"]["save_dir"]
    model_name = config["training"]["model_name"]
    checkpoint_path = os.path.join(save_dir, f"best_{model_name}.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Please run train.py first.")
        return

    # Load DataLoaders
    _, _, test_loader, superclasses = get_dataloaders(config)

    # Rebuild & Load Model Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name, in_channels=12, num_classes=len(superclasses))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\nEvaluating saved model (Epoch {checkpoint['epoch']}, Val AUC {checkpoint['val_auc']:.4f}) on Test Set (Fold 10)...")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for signals, targets in test_loader:
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate Metrics per Superclass
    metrics_list = []
    binary_preds = (all_preds >= 0.5).astype(int)

    for i, sclass in enumerate(superclasses):
        auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        f1 = f1_score(all_targets[:, i], binary_preds[:, i], zero_division=0)
        prec = precision_score(all_targets[:, i], binary_preds[:, i], zero_division=0)
        rec = recall_score(all_targets[:, i], binary_preds[:, i], zero_division=0)

        metrics_list.append({
            "Superclass": sclass,
            "ROC-AUC": auc,
            "F1-Score": f1,
            "Precision": prec,
            "Recall": rec,
            "Support": int(all_targets[:, i].sum())
        })

    metrics_df = pd.DataFrame(metrics_list)
    macro_auc = roc_auc_score(all_targets, all_preds, average="macro")
    macro_f1 = f1_score(all_targets, binary_preds, average="macro", zero_division=0)

    print("\n=======================================================")
    print(f"       TEST SET EVALUATION RESULTS ({model_name})       ")
    print("=======================================================")
    print(metrics_df.to_string(index=False))
    print("-------------------------------------------------------")
    print(f"Overall Macro ROC-AUC: {macro_auc:.4f}")
    print(f"Overall Macro F1-Score: {macro_f1:.4f}")
    print("=======================================================")

    # Plot & Save ROC Curves
    plot_roc_curves(all_targets, all_preds, superclasses, os.path.join(save_dir, "test_roc_curves.png"))

    # Save metrics table to CSV
    metrics_csv_path = os.path.join(save_dir, "test_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved evaluation metrics report to {metrics_csv_path}")


if __name__ == "__main__":
    evaluate_test_set()
