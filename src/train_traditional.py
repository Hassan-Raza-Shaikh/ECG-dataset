import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.dataset import compute_superclasses, SUPERCLASSES
from src.features import extract_dataset_features


def get_traditional_models():
    models = {
        "RandomForest": MultiOutputClassifier(
            RandomForestClassifier(n_estimators=150, max_depth=15, n_jobs=-1, random_state=42)
        ),
        "ExtraTrees": MultiOutputClassifier(
            ExtraTreesClassifier(n_estimators=150, max_depth=15, n_jobs=-1, random_state=42)
        ),
        "HistGradientBoosting": MultiOutputClassifier(
            HistGradientBoostingClassifier(max_iter=150, random_state=42)
        ),
        "SVM_Linear": MultiOutputClassifier(
            CalibratedClassifierCV(LinearSVC(C=0.1, dual=False, random_state=42, max_iter=2000))
        ),
        "LogisticRegression": MultiOutputClassifier(
            LogisticRegression(C=1.0, max_iter=500, random_state=42)
        ),
    }
    return models


def train_evaluate_traditional_models(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    db_path = config["data"]["database_csv"]
    scp_path = config["data"]["scp_csv"]
    sampling_rate = config["data"]["sampling_rate"]
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load metadata
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = pd.read_csv(scp_path, index_col=0)
    df = compute_superclasses(df, scp_df)

    # 2. Load cached signal matrix
    cache_path = os.path.join(raw_dir, f"signals_cache_{sampling_rate}hz.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Signal cache {cache_path} not found.")

    all_signals = np.load(cache_path)

    # 3. Extract Features
    X_features = extract_dataset_features(all_signals, sampling_rate=sampling_rate, raw_dir=raw_dir)
    y_all = df[SUPERCLASSES].values

    # 4. Split dataset (Fold 1-8 Train, Fold 9 Val, Fold 10 Test)
    train_idx = df.strat_fold.isin(config["split"]["train_folds"]).values
    val_idx = df.strat_fold.isin(config["split"]["val_folds"]).values
    test_idx = df.strat_fold.isin(config["split"]["test_folds"]).values

    # Fit StandardScaler on train set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_features[train_idx])
    X_val = scaler.transform(X_features[val_idx])
    X_test = scaler.transform(X_features[test_idx])

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    # Save scaler
    scaler_path = os.path.join(save_dir, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # 5. Train & Evaluate Traditional Models
    models = get_traditional_models()
    results = []
    test_predictions = {}

    print(f"\n--- Training {len(models)} Traditional ML & Ensemble Models ---", flush=True)

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(save_dir, f"{name}.joblib")
        joblib.dump(model, model_path)

        # Predict probabilities on Validation and Test sets
        val_probs_list = model.predict_proba(X_val) # List of (N, 2) arrays per output
        val_probs = np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel() for p in val_probs_list])

        test_probs_list = model.predict_proba(X_test)
        test_probs = np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel() for p in test_probs_list])

        test_predictions[name] = test_probs

        val_auc = roc_auc_score(y_val, val_probs, average="macro")
        test_auc = roc_auc_score(y_test, test_probs, average="macro")

        binary_preds = (test_probs >= 0.5).astype(int)
        test_f1 = f1_score(y_test, binary_preds, average="macro", zero_division=0)
        test_prec = precision_score(y_test, binary_preds, average="macro", zero_division=0)
        test_rec = recall_score(y_test, binary_preds, average="macro", zero_division=0)

        print(f"   [{name}] Val Macro AUC: {val_auc:.4f} | Test Macro AUC: {test_auc:.4f} | Test Macro F1: {test_f1:.4f}", flush=True)

        results.append({
            "Model": name,
            "Val Macro AUC": val_auc,
            "Test Macro AUC": test_auc,
            "Test Macro F1": test_f1,
            "Test Precision": test_prec,
            "Test Recall": test_rec,
        })

    results_df = pd.DataFrame(results).sort_values(by="Test Macro AUC", ascending=False)
    print("\n=======================================================")
    print("      TRADITIONAL ML & ENSEMBLE TEST BENCHMARK        ")
    print("=======================================================")
    print(results_df.to_string(index=False))
    print("=======================================================")

    # Save test predictions for ensembling
    np.savez(os.path.join(save_dir, "traditional_test_preds.npz"), **test_predictions)
    return results_df


if __name__ == "__main__":
    train_evaluate_traditional_models()
