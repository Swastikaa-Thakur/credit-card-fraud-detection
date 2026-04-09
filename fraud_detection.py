"""
Credit Card Fraud Detection using Random Forest
Author: Swastika
Description: Detects fraudulent credit card transactions using
             a Random Forest classifier, evaluated with precision & recall.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import warnings
warnings.filterwarnings("ignore")
 
# ──────────────────────────────────────────────────────────
#  CONFIGURATION  (change these if needed)
# ──────────────────────────────────────────────────────────
DATASET_PATH   = "creditcard.csv"   # path to Kaggle CSV
USE_SYNTHETIC  = not os.path.exists(DATASET_PATH)  # auto-fallback
N_SAMPLES      = 10_000             # used only for synthetic data
FRAUD_RATIO    = 0.02               # used only for synthetic data
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
 
 
# ══════════════════════════════════════════════════════════
#  1.  LOAD  DATA
# ══════════════════════════════════════════════════════════
def load_data():
    """
    Load real Kaggle dataset if available, otherwise generate
    a synthetic dataset that mimics its structure.
    """
    if not USE_SYNTHETIC:
        print(f"[✔] Loading real dataset from '{DATASET_PATH}' ...")
        df = pd.read_csv(DATASET_PATH)
 
        # Kaggle dataset has 'Time' column — drop it (not useful here)
        if "Time" in df.columns:
            df.drop(columns=["Time"], inplace=True)
 
        print(f"    Rows: {len(df):,}  |  Columns: {df.shape[1]}")
        print(f"    Fraudulent transactions : {df['Class'].sum():,} "
              f"({df['Class'].mean()*100:.3f}%)")
        return df
 
    else:
        print("[!] 'creditcard.csv' not found — using SYNTHETIC dataset.")
        print("    Download real data: "
              "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n")
        return _generate_synthetic()
 
 
def _generate_synthetic():
    """Generate a synthetic credit card transaction dataset."""
    np.random.seed(RANDOM_STATE)
    n_fraud = int(N_SAMPLES * FRAUD_RATIO)
    n_legit = N_SAMPLES - n_fraud
 
    # Legitimate transactions
    legit = pd.DataFrame({
        "Amount": np.random.exponential(scale=80,  size=n_legit),
        "Hour":   np.random.randint(0, 24,          size=n_legit),
        **{f"V{i}": np.random.normal(0, 1, n_legit) for i in range(1, 8)},
        "Class":  0,
    })
 
    # Fraudulent transactions — different distribution
    fraud = pd.DataFrame({
        "Amount": np.random.exponential(scale=200, size=n_fraud),
        "Hour":   np.random.choice([0,1,2,3,22,23], size=n_fraud),
        "V1":     np.random.normal(-3.0,  1.5, n_fraud),
        "V2":     np.random.normal( 2.0,  1.5, n_fraud),
        "V3":     np.random.normal(-2.0,  1.5, n_fraud),
        "V4":     np.random.normal( 1.5,  1.0, n_fraud),
        "V5":     np.random.normal(-1.0,  1.0, n_fraud),
        "V6":     np.random.normal( 0.5,  1.0, n_fraud),
        "V7":     np.random.normal(-0.5,  1.0, n_fraud),
        "Class":  1,
    })
 
    df = (pd.concat([legit, fraud], ignore_index=True)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True))
 
    print(f"    Synthetic rows : {len(df):,}")
    print(f"    Fraudulent     : {df['Class'].sum():,} "
          f"({df['Class'].mean()*100:.2f}%)\n")
    return df
 
 
# ══════════════════════════════════════════════════════════
#  2.  EXPLORE  DATA
# ══════════════════════════════════════════════════════════
def explore_data(df):
    """Print basic statistics and class distribution."""
    print("\n" + "─"*50)
    print("  DATA OVERVIEW")
    print("─"*50)
    print(df.head(3).to_string())
    print(f"\nShape   : {df.shape}")
    print(f"Missing : {df.isnull().sum().sum()} values")
 
    counts = df["Class"].value_counts()
    print(f"\nClass distribution:")
    print(f"  Legitimate (0) : {counts[0]:>7,}")
    print(f"  Fraudulent (1) : {counts[1]:>7,}")
    print("─"*50)
 
 
# ══════════════════════════════════════════════════════════
#  3.  PREPROCESS
# ══════════════════════════════════════════════════════════
def preprocess(df):
    """Scale Amount, split X / y, train/test split."""
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
 
    X = df.drop("Class", axis=1)
    y = df["Class"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
 
    print(f"\n[✔] Train size : {len(X_train):,}")
    print(f"    Test  size : {len(X_test):,}")
    return X_train, X_test, y_train, y_test
 
 
# ══════════════════════════════════════════════════════════
#  4.  TRAIN  MODEL
# ══════════════════════════════════════════════════════════
def train_model(X_train, y_train):
    """Train a Random Forest with balanced class weights."""
    print("\n[⏳] Training Random Forest ...")
 
    model = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 10,
        min_samples_split = 5,
        class_weight  = "balanced",   # handles class imbalance
        random_state  = RANDOM_STATE,
        n_jobs        = -1,           # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("[✔] Training complete!\n")
    return model
 
 
# ══════════════════════════════════════════════════════════
#  5.  EVALUATE  MODEL
# ══════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test):
    """Print all metrics and save evaluation plots."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
 
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)
 
    # ── Console output ──
    print("=" * 52)
    print("         MODEL EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Precision  : {precision:.4f}   "
          "(how many predicted frauds are real)")
    print(f"  Recall     : {recall:.4f}   "
          "(how many actual frauds were caught)")
    print(f"  F1-Score   : {f1:.4f}   "
          "(balance of precision & recall)")
    print(f"  ROC-AUC    : {roc_auc:.4f}   "
          "(overall discrimination ability)")
    print("=" * 52)
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Fraud"]
    ))
 
    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Credit Card Fraud Detection — Model Evaluation",
        fontsize=14, fontweight="bold"
    )
 
    # Plot 1 — Confusion Matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold")
 
    # Plot 2 — ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"AUC = {roc_auc:.3f}")
    axes[1].fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve", fontweight="bold")
    axes[1].legend(loc="lower right")
 
    # Plot 3 — Precision vs Recall bar
    metrics = {"Precision": precision, "Recall": recall,
               "F1-Score": f1, "ROC-AUC": roc_auc}
    bars = axes[2].bar(
        metrics.keys(), metrics.values(),
        color=["#4f8ef7", "#f7854f", "#4ff7a7", "#b04ff7"],
        edgecolor="white", width=0.5
    )
    axes[2].set_ylim(0, 1.15)
    axes[2].set_title("Metrics Summary", fontweight="bold")
    for bar, val in zip(bars, metrics.values()):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10
        )
 
    plt.tight_layout()
    plt.savefig("evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✔] Saved → evaluation_plots.png")
 
    return y_pred
 
 
# ══════════════════════════════════════════════════════════
#  6.  FEATURE  IMPORTANCE
# ══════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_names):
    """Horizontal bar chart of top feature importances."""
    importances = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
        .tail(15)          # show top 15 features
    )
 
    colors = plt.cm.Blues(
        np.linspace(0.4, 0.9, len(importances))
    )
 
    plt.figure(figsize=(9, 6))
    bars = plt.barh(importances.index, importances.values,
                    color=colors, edgecolor="white")
    plt.title("Top Feature Importances — Random Forest",
              fontweight="bold", fontsize=13)
    plt.xlabel("Importance Score")
    for bar, val in zip(bars, importances.values):
        plt.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✔] Saved → feature_importance.png")
 
 
# ══════════════════════════════════════════════════════════
#  7.  CLASS  DISTRIBUTION  PLOT
# ══════════════════════════════════════════════════════════
def plot_class_distribution(df):
    """Bar chart showing class imbalance."""
    counts = df["Class"].value_counts()
    labels = ["Legitimate", "Fraudulent"]
    colors = ["#4f8ef7", "#f74f4f"]
 
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts.values, color=colors,
                   edgecolor="white", width=0.4)
    plt.title("Class Distribution", fontweight="bold")
    plt.ylabel("Number of Transactions")
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50,
                 f"{val:,}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✔] Saved → class_distribution.png")
 
 
# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
 
    print("\n" + "╔" + "═"*50 + "╗")
    print("║   CREDIT CARD FRAUD DETECTION SYSTEM" + " "*13 + "║")
    print("╚" + "═"*50 + "╝\n")
 
    # Step 1 — Load
    print("[1/6] Loading data ...")
    df = load_data()
 
    # Step 2 — Explore
    print("[2/6] Exploring data ...")
    explore_data(df)
 
    # Step 3 — Class distribution plot
    print("\n[3/6] Plotting class distribution ...")
    plot_class_distribution(df)
 
    # Step 4 — Preprocess
    print("\n[4/6] Preprocessing ...")
    X_train, X_test, y_train, y_test = preprocess(df)
 
    # Step 5 — Train
    print("\n[5/6] Training model ...")
    model = train_model(X_train, y_train)
 
    # Step 6 — Evaluate
    print("[6/6] Evaluating model ...")
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_train.columns.tolist())
 
    print("\n✅ All done!")
    print("   Output files:")
    print("   • evaluation_plots.png")
    print("   • feature_importance.png")
    print("   • class_distribution.png\n")
