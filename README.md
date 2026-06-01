# 🔍 Credit Card Fraud Detection

End-to-end ML pipeline for detecting fraudulent credit card transactions using Random Forest and XGBoost, with SHAP explainability.

---

## 📊 Dataset

- **284,807** credit card transactions
- Severe class imbalance — only **0.17%** are fraud
- Features `V1`–`V28` are PCA-transformed (pre-scaled), plus `Time`, `Amount`, and `Class`

---

## 🔧 Pipeline Overview

| Step | Description |
|------|-------------|
| **EDA** | Explored class distribution and feature statistics |
| **Feature Engineering** | Converted `Time` → `Hour`; log-transformed `Amount` |
| **Train/Test Split** | 80/20 stratified split to preserve fraud ratio |
| **Scaling** | `RobustScaler` applied to `Hour` and `Log_Amount` |
| **Class Imbalance** | SMOTE oversampling on training data only |
| **Modelling** | Random Forest (baseline) + XGBoost (main model) |
| **Threshold Tuning** | Optimised decision threshold using F1 score |
| **Business Cost** | Translated model errors into dollar impact |
| **Explainability** | SHAP values for global and per-transaction insights |

---

## 🤖 Models

### Random Forest (Baseline)
- 100 estimators, max depth 14
- Trained on SMOTE-resampled data
- `class_weight='balanced'`

### XGBoost (Main Model)
- 1,500 estimators with early stopping (100 rounds)
- `scale_pos_weight` set to genuine/fraud ratio (~579) instead of SMOTE
- Monitored with PR-AUC during training
- **Best PR-AUC: 0.8804**

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| PR-AUC | 0.8804 |
| ROC-AUC | — |
| Optimised threshold | Best F1 |

> PR-AUC is the primary metric — it is more meaningful than accuracy or ROC-AUC on heavily imbalanced data.

---

## 💰 Business Cost Estimate

| Error Type | Cost | Description |
|---|---|---|
| False Negative (missed fraud) | $200 | Undetected fraud loss |
| False Positive (false alarm) | $35 | Manual review cost |

---

## 🧠 Explainability (SHAP)

- **Global importance**: Bar chart of mean absolute SHAP values across 5,000 sampled transactions
- **Per-transaction**: Waterfall plots comparing a flagged fraud vs. a genuine transaction
- **Feature interaction**: Scatter plot showing how `V14` (top feature) drives fraud probability
- Top driver: **`V14`**

---

## 🗂 Saved Artefacts

| File | Description |
|------|-------------|
| `fraud_model.joblib` | Trained XGBoost model |
| `scaler.joblib` | Fitted RobustScaler |
| `model_config.json` | Threshold, base value, feature names, metadata |

---

## ⚙️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

```python
import joblib, json
import numpy as np

model  = joblib.load('fraud_model.joblib')
scaler = joblib.load('scaler.joblib')

with open('model_config.json') as f:
    config = json.load(f)

THRESHOLD = config['threshold']

# Scale your new transaction, then:
proba = model.predict_proba(X_new)[:, 1]
prediction = (proba >= THRESHOLD).astype(int)
```
