"""
Credit Card Fraud Detection using Random Forest
Author: Swastika
Description: Detects fraudulent credit card transactions using
             a Random Forest classifier, evaluated with precision & recall.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
def generate_dataset(n_samples=10000, fraud_ratio=0.02, random_state=42):
    """Generate a synthetic credit card transaction dataset."""
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        "Amount":       np.random.exponential(scale=80, size=n_legit),
        "Hour":         np.random.randint(0, 24, n_legit),
        "V1":           np.random.normal(0, 1, n_legit),
        "V2":           np.random.normal(0, 1, n_legit),
        "V3":           np.random.normal(0, 1, n_legit),
        "V4":           np.random.normal(0, 1, n_legit),
        "V5":           np.random.normal(0, 1, n_legit),
        "Class":        0
    })

    # Fraudulent transactions (different distribution)
    fraud = pd.DataFrame({
        "Amount":       np.random.exponential(scale=200, size=n_fraud),
        "Hour":         np.random.choice([0,1,2,3,22,23], size=n_fraud),   # late night
        "V1":           np.random.normal(-3, 1.5, n_fraud),
        "V2":           np.random.normal(2, 1.5, n_fraud),
        "V3":           np.random.normal(-2, 1.5, n_fraud),
        "V4":           np.random.normal(1.5, 1, n_fraud),
        "V5":           np.random.normal(-1, 1, n_fraud),
        "Class":        1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=random_state)
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    """Scale Amount and split features/labels."""
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


# ─────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# 4. EVALUATE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """Print metrics and generate evaluation plots."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    print("\n" + "="*50)
    print("       MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # ── Plot 1: Confusion Matrix ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Credit Card Fraud Detection – Model Evaluation", fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    # ── Plot 2: ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0,1],[0,1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[✔] Plots saved as 'evaluation_plots.png'")

    return y_pred


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names):
    """Bar chart of feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="steelblue", edgecolor="white")
    plt.title("Feature Importances – Random Forest", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✔] Feature importance plot saved as 'feature_importance.png'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("[1/4] Generating synthetic dataset...")
    df = generate_dataset()
    print(f"      Total transactions : {len(df)}")
    print(f"      Fraudulent          : {df['Class'].sum()} ({df['Class'].mean()*100:.1f}%)")

    print("\n[2/4] Preprocessing data...")
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[3/4] Training Random Forest model...")
    model = train_model(X_train, y_train)
    print("      Training complete!")

    print("\n[4/4] Evaluating model...")
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X.columns.tolist())

    print("\n✅ Done! Check evaluation_plots.png and feature_importance.png")
