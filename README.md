# Credit Card Fraud Detection – Machine Learning Project

## Overview
This project builds a supervised machine learning pipeline to detect fraudulent credit card transactions using the popular **Credit Card Fraud Detection** dataset.  
The goal is to compare multiple ML models, handle severe class imbalance, perform EDA, and evaluate models with metrics appropriate for fraud detection.

This repository contains:
- A complete Jupyter Notebook with EDA, preprocessing, modeling, tuning, and evaluation.
- Model comparison using ROC-AUC and Precision–Recall curves.
- A structured ML workflow that can be extended or deployed.

---

## Dataset
The dataset is publicly available on Kaggle:

🔗 **https://www.kaggle.com/mlg-ulb/creditcardfraud**

It contains:
- **284,807 transactions**
- **492 fraud cases (~0.17%)**
- PCA-transformed features: `V1`–`V28`
- Original numerical features: `Amount`, `Time`
- Target variable:  
  - `0` → Legitimate  
  - `1` → Fraudulent  

Place the file `creditcard.csv` inside the `data/` folder.

---

## Project Structure

```
project/
│
├── data/
│   └── creditcard.csv       # dataset (not included in repo)
│
├── credit_card_fraud_detection_project.ipynb
│
├── README.md                # this file
│
└── requirements.txt         # package dependencies
```

---

## Methods & Models

### ✔ Exploratory Data Analysis
- Histograms, KDE, boxplots
- Fraud class imbalance visualization
- Correlation heatmap (PCA components)

### ✔ Data Preprocessing
- Train/test split with stratification
- Scaling with StandardScaler via Pipeline
- Handling class imbalance using:
  - `class_weight='balanced'`
  - No resampling (SMOTE optional for future work)

### ✔ Models Implemented
- **Logistic Regression**
- **Random Forest Classifier**
- **Tuned Logistic Regression (GridSearchCV)**

---

## Evaluation Metrics

Because the dataset is extremely imbalanced, we focus on:

- **Recall** (catching fraud is critical)
- **Precision** (avoid too many false alarms)
- **F1-score**
- **ROC-AUC**
- **Average Precision (AP)**
- **Confusion Matrix**
- **Precision–Recall Curve**

Accuracy is *not meaningful* for this dataset.

---

## Results Summary

| Model | ROC-AUC | Average Precision | Notes |
|-------|---------|-------------------|-------|
| Logistic Regression | good | moderate | Interpretable, linear |
| Random Forest | higher | higher | Handles imbalance better |
| Tuned Logistic Regression | improved | moderate | Benefit from hyperparameter tuning |

Random Forest typically performs the strongest.

---

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Download dataset from Kaggle and place here:
```
project/data/creditcard.csv
```

### 3. Run the notebook
```
jupyter notebook credit_card_fraud_detection_project.ipynb
```

---

## Future Improvements
- Use SMOTE / ADASYN for synthetic oversampling  
- Train more advanced models: XGBoost, LightGBM  
- Deploy a real-time fraud scoring API  
- Use anomaly detection or unsupervised techniques  
- Investigate cost-sensitive learning (vary penalties for fraud)

---

## Author
This notebook was generated with assistance from ChatGPT and can be used as a capstone project for supervised machine learning.

