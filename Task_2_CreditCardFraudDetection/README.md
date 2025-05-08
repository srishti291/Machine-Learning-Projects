# Task 2 – Credit Card Fraud Detection

## 📌 Objective
Build a machine learning model to detect fraudulent credit card transactions using a realistic dataset with class imbalance.

---

## 📊 Dataset
- Source: `fraudTest.csv` from Kaggle's fraud detection dataset
- Total sample used: 10,000 records
  - 1,000 fraud
  - 9,000 non-fraud

## 📁 Dataset Note
The original dataset `fraudTest.csv` was too large (142 MB) to be uploaded to GitHub, which has a 100 MB limit.

To test the code, please use a sample or smaller version of the dataset with the same structure, or contact me if needed.


---

## 🔍 Features Used
Selected features after cleaning:
- `amt`, `gender`, `category`, `job`, `state`, `zip`, `lat`, `long`, `merch_lat`, `merch_long`, etc.
- Removed high-cardinality and non-numeric columns like `cc_num`, `merchant`, `first`, `last`, etc.

---

## ⚙️ Preprocessing
- One-hot encoding on selected categorical columns (`gender`, `category`, `job`, `state`)
- Standardization using `StandardScaler`
- Balanced data using **SMOTE**

---

## 🧠 Model Used
- **RandomForestClassifier** from scikit-learn
- `class_weight` handled via SMOTE instead of weighting
- Train-test split: 80/20

---

## 📈 Results

| Metric        | Class 0 (Non-Fraud) | Class 1 (Fraud) |
|---------------|---------------------|-----------------|
| Precision     | 0.98                | 0.99            |
| Recall        | 0.99                | 0.98            |
| F1-score      | 0.98                | 0.98            |
| Accuracy      | **98%** overall     |                 |

Confusion Matrix:
[[1800 26]
[ 30 1744]]

---

##  Files
- `main.py` – Full model code
- `results.txt` – Evaluation output
- `fraudTest.csv` – Input dataset
- `README.md` – Project summary

---

## Author
Srishti Gupta  
CodSoft Internship – Machine Learning Track  
Task 2 Completed ✅
