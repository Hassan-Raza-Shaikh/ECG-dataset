import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve

from src.dataset import get_dataloaders
from src.models import build_model
from src.train import get_device, evaluate_model, train_pipeline
from src.train_traditional import train_evaluate_traditional_models


def get_deep_model_predictions(model_name, dataloader, device, num_classes=5):
    """
    Loads saved deep learning checkpoint and outputs predicted test probabilities.
    """
    save_dir = "artifacts/models"
    checkpoint_path = os.path.join(save_dir, f"best_{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name, in_channels=12, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for signals, targets in dataloader:
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.numpy())

    return np.vstack(all_preds), np.vstack(all_targets)


def plot_multi_model_auc_bar(df_results, save_path):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    df_sorted = df_results.sort_values(by="Test Macro AUC", ascending=True)
    palette = ["#e76f51" if "Ensemble" in m else "#2a9d8f" if "ResNet" in m or "CNN" in m else "#457b9d" for m in df_sorted["Model"]]

    ax = sns.barplot(x="Test Macro AUC", y="Model", data=df_sorted, palette=palette)
    plt.title("Multi-Model ECG Benchmark Comparison (Test Set Fold 10)", fontsize=15, fontweight="bold")
    plt.xlabel("Test Macro ROC-AUC", fontsize=12)
    plt.ylabel("Model Architecture / Ensemble", fontsize=12)
    plt.xlim([0.5, 1.0])

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved multi-model AUC comparison bar chart to {save_path}", flush=True)


def plot_multi_model_roc_curves(predictions_dict, y_test, save_path):
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    for name, probs in predictions_dict.items():
        macro_auc = roc_auc_score(y_test, probs, average="macro")
        # Micro-average ROC curve
        fpr, tpr, _ = roc_curve(y_test.ravel(), probs.ravel())
        linewidth = 2.5 if "Ensemble" in name or "ResNet" in name else 1.5
        linestyle = "-" if "Ensemble" in name else "--" if "ResNet" in name else ":"
        plt.plot(fpr, tpr, label=f"{name} (Macro AUC = {macro_auc:.4f})", linewidth=linewidth, linestyle=linestyle)

    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Multi-Model ROC Curves Comparison (Test Set Fold 10)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved multi-model ROC overlay curves to {save_path}", flush=True)


def run_full_benchmark(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1. Train Traditional ML & Ensemble Models
    print("\n=======================================================", flush=True)
    print(" STEP 1: TRAINING TRADITIONAL ML & TREE ENSEMBLE MODELS ", flush=True)
    print("=======================================================", flush=True)
    trad_results_df = train_evaluate_traditional_models(config_path)

    # 2. Train Deep Learning Architecture 2 (CNN-BiLSTM-Attention)
    print("\n=======================================================", flush=True)
    print(" STEP 2: TRAINING DEEP LEARNING ARCHITECTURE (CNN-BiLSTM)", flush=True)
    print("=======================================================", flush=True)
    config_bilstm = config.copy()
    config_bilstm["training"]["model_name"] = "CNN_BiLSTM_Attention"
    config_bilstm["training"]["epochs"] = 5
    
    # Save temporary config and run training
    temp_cfg_path = "configs/config_bilstm.yaml"
    with open(temp_cfg_path, "w") as f:
        yaml.dump(config_bilstm, f)

    train_pipeline(temp_cfg_path)
    if os.path.exists(temp_cfg_path):
        os.remove(temp_cfg_path)

    # 3. Load All Model Predictions on Test Set (Fold 10)
    print("\n=======================================================", flush=True)
    print(" STEP 3: EVALUATING ALL MODELS & BUILDING STACKING ENSEMBLE", flush=True)
    print("=======================================================", flush=True)

    _, _, test_loader, superclasses = get_dataloaders(config)

    # Predictions dictionary: model_name -> test_probs (N, 5)
    test_preds_dict = {}

    # Load Deep Learning Models
    resnet_probs, y_test = get_deep_model_predictions("ResNet1D", test_loader, device)
    if resnet_probs is not None:
        test_preds_dict["ResNet1D"] = resnet_probs

    bilstm_probs, _ = get_deep_model_predictions("CNN_BiLSTM_Attention", test_loader, device)
    if bilstm_probs is not None:
        test_preds_dict["CNN_BiLSTM_Attention"] = bilstm_probs

    # Load Traditional ML Predictions
    trad_preds_file = os.path.join(save_dir, "traditional_test_preds.npz")
    if os.path.exists(trad_preds_file):
        trad_data = np.load(trad_preds_file)
        for name in trad_data.files:
            test_preds_dict[name] = trad_data[name]

    # 4. Build Weighted Stacking Ensemble
    # Blend: 50% ResNet1D + 30% CNN-BiLSTM-Attention + 10% RandomForest + 10% ExtraTrees
    ensemble_weights = {
        "ResNet1D": 0.50,
        "CNN_BiLSTM_Attention": 0.30,
        "RandomForest": 0.10,
        "ExtraTrees": 0.10
    }

    ensemble_probs = np.zeros_like(y_test, dtype=np.float32)
    weight_sum = 0.0

    for model_name, weight in ensemble_weights.items():
        if model_name in test_preds_dict:
            ensemble_probs += weight * test_preds_dict[model_name]
            weight_sum += weight

    if weight_sum > 0:
        ensemble_probs /= weight_sum
        test_preds_dict["Weighted_Stacking_Ensemble"] = ensemble_probs

    # 5. Calculate Comprehensive Benchmark Table
    final_benchmark_list = []
    for name, probs in test_preds_dict.items():
        macro_auc = roc_auc_score(y_test, probs, average="macro")
        binary_preds = (probs >= 0.5).astype(int)
        macro_f1 = f1_score(y_test, binary_preds, average="macro", zero_division=0)
        prec = precision_score(y_test, binary_preds, average="macro", zero_division=0)
        rec = recall_score(y_test, binary_preds, average="macro", zero_division=0)

        final_benchmark_list.append({
            "Model": name,
            "Test Macro AUC": macro_auc,
            "Test Macro F1": macro_f1,
            "Test Precision": prec,
            "Test Recall": rec
        })

    benchmark_df = pd.DataFrame(final_benchmark_list).sort_values(by="Test Macro AUC", ascending=False)

    print("\n=========================================================================")
    print("         COMPLETE MULTI-MODEL ECG BENCHMARK & ENSEMBLE RESULTS           ")
    print("=========================================================================")
    print(benchmark_df.to_string(index=False))
    print("=========================================================================")

    # Save CSV benchmark report
    benchmark_csv = os.path.join(save_dir, "multi_model_benchmark_results.csv")
    benchmark_df.to_csv(benchmark_csv, index=False)
    print(f"Saved benchmark results table to {benchmark_csv}", flush=True)

    # 6. Plot & Save Visual Comparison Artifacts
    plot_multi_model_auc_bar(benchmark_df, os.path.join(save_dir, "model_auc_comparison_bar.png"))
    plot_multi_model_roc_curves(test_preds_dict, y_test, os.path.join(save_dir, "multi_model_roc_curves.png"))

    print("\nAll multi-model comparisons and ensemble evaluations completed successfully!")


if __name__ == "__main__":
    run_full_benchmark()
