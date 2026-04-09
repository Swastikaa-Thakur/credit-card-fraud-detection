# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a **Random Forest** classifier, evaluated with **Precision**, **Recall**, and **ROC-AUC** metrics.

---

## 📌 Features

- Synthetic credit card transaction dataset (customizable size & fraud ratio)
- Data preprocessing with StandardScaler
- Random Forest model with `class_weight="balanced"` to handle imbalanced data
- Evaluation: Precision, Recall, F1-Score, ROC-AUC
- Visual outputs: Confusion Matrix, ROC Curve, Feature Importance chart

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.py     # Main script
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python fraud_detection.py
```

---

## 📊 Sample Output

```
[1/4] Generating synthetic dataset...
      Total transactions : 10000
      Fraudulent          : 200 (2.0%)

[2/4] Preprocessing data...

[3/4] Training Random Forest model...
      Training complete!

[4/4] Evaluating model...

==================================================
       MODEL EVALUATION RESULTS
==================================================
  Precision  : 0.9200
  Recall     : 0.8750
  F1-Score   : 0.8969
  ROC-AUC    : 0.9812
==================================================
```

---

## 🧠 Key Concepts

| Term | Description |
|------|-------------|
| **Precision** | Of all predicted frauds, how many were actually fraud |
| **Recall** | Of all actual frauds, how many were correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Model's ability to distinguish between classes |

---

## 🛠️ Tech Stack

- Python 3.8+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

---

## 👩‍💻 Author

**Swastika** — [GitHub](https://github.com/your-username)
