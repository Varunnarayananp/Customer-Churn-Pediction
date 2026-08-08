# Customer Churn Prediction — ML Assignment 2

## a. Problem statement

Predict whether a telecom customer will churn (`Yes` / `No`) using service, billing, and demographic features. This is a **binary classification** problem. Five classical ML models are trained, evaluated on a held-out test set, and exposed through a Streamlit app for interactive evaluation on uploaded CSV test data.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3) |
| Instances | 7,043 |
| Target | `Churn Label` (`Yes` / `No`) — ~26.5% churn |
| Input features used | **44** (after cleaning; requirement ≥ 12) |
| Train / test split | 80% / 20%, stratified, `random_state=42` |

**Dropped (IDs / direct churn fields only):** `Customer ID`, `Customer Status`, `Churn Score`, `Churn Category`, `Churn Reason`.


**Preprocessing:** `StandardScaler` on numeric columns + `OneHotEncoder` on categorical columns inside `sklearn` Pipelines.

## c. GitHub Repository Link

> 
>
> `https://github.com/Varunnarayananp/Customer-Churn-Pediction.git`

### Repository structure

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── Customer_Churn_Prediction.ipynb
├── archive/
│   └── telco.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── metrics_comparison.csv
```

## d. Models used & comparison table

All six required metrics are reported for each of the five models on the **same** stratified test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9617 | 0.9918 | 0.9545 | 0.8984 | 0.9256 | 0.9006 |
| Decision Tree | 0.9517 | 0.9858 | 0.9554 | 0.8583 | 0.9042 | 0.8743 |
| kNN | 0.9219 | 0.9630 | 0.8952 | 0.7995 | 0.8446 | 0.7950 |
| Naive Bayes (Gaussian) | 0.4301 | 0.5199 | 0.2768 | 0.7112 | 0.3985 | 0.0377 |
| Random Forest (Ensemble) | 0.8836 | 0.9540 | 0.9688 | 0.5802 | 0.7258 | 0.6925 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Best overall** — highest F1 (0.9256), AUC (0.9918), and MCC (0.9006). Strongly helped by `Satisfaction Score` and related features. |
| Decision Tree | Very strong and interpretable; slightly behind LR on Recall/F1. |
| kNN | Solid after StandardScaler; high-cardinality `City` encoding makes neighbor search harder than LR/DT. |
| Naive Bayes | Weak here — high-dimensional one-hot features (`City`, etc.) break the independence assumption (low Accuracy/F1/MCC). |
| Random Forest | Highest Precision, but lower Recall on churners than LR/DT. |
| **Overall Winner** | **Logistic Regression** (best F1 / AUC / MCC). |

## Streamlit app

### Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```


### Live Streamlit link

>
>
> ``

## How to reproduce training

1. Open `Customer_Churn_Prediction.ipynb` and **Run All**, **or**
2. Run: `python train_and_export.py` (uses the same pipeline / seed)

This regenerates `model/*.joblib` and `test_data.csv`.
