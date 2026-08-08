# Customer Churn Prediction — ML Assignment 2

## a. Problem statement

Predict whether a telecom customer will churn (`Yes` / `No`) using service, billing, and demographic features. This is a **binary classification** problem. Five classical ML models are trained, evaluated on a held-out test set, and exposed through a Streamlit app for interactive evaluation on uploaded CSV test data.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3) |
| Instances | 7,043 |
| Target | `Churn Label` (`Yes` / `No`) — ~26.5% churn |
| Input features used | **36** (after cleaning; requirement ≥ 12) |
| Train / test split | 80% / 20%, stratified, `random_state=42` |

**Dropped (IDs / constants / leakage):** `Customer ID`, `Country`, `State`, `Quarter`, `City`, `Zip Code`, `Latitude`, `Longitude`, `Customer Status`, `Churn Score`, `Churn Category`, `Churn Reason`, `Satisfaction Score`.

**Null handling:** `Offer` and `Internet Type` filled with `None`.

**Preprocessing:** `StandardScaler` on numeric columns + `OneHotEncoder` on categorical columns inside `sklearn` Pipelines.

## c. GitHub Repository Link

> Replace this line with your live GitHub repo URL after you push:
>
> `https://github.com/<your-username>/<your-repo-name>`

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
| Logistic Regression | 0.8502 | 0.9100 | 0.7390 | 0.6738 | 0.7049 | 0.6060 |
| Decision Tree | 0.8368 | 0.8789 | 0.7130 | 0.6444 | 0.6770 | 0.5694 |
| kNN | 0.8304 | 0.8760 | 0.6891 | 0.6578 | 0.6731 | 0.5589 |
| Naive Bayes (Gaussian) | 0.7800 | 0.8757 | 0.5611 | 0.7861 | 0.6548 | 0.5152 |
| Random Forest (Ensemble) | 0.8502 | 0.9036 | 0.7801 | 0.6070 | 0.6827 | 0.5946 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall on this test set: highest **F1 (0.7049)**, **AUC (0.9100)**, and **MCC (0.6060)**. Balanced Precision/Recall after scaling + one-hot encoding. |
| Decision Tree | Interpretable rules; depth/leaf constraints limit overfit, but trails LR/RF slightly on F1 and AUC. |
| kNN | Competitive after StandardScaler; still affected by imbalance and high-dimensional encoded space. |
| Naive Bayes | Highest **Recall (0.7861)** but lowest Precision — catches more churners at the cost of false alarms. Independence assumption hurts F1/MCC. |
| Random Forest | Tied-best Accuracy and strong Precision/AUC; slightly lower Recall/F1 than Logistic Regression on the churn class. |
| **Overall Winner** | **Logistic Regression** (best F1 / AUC / MCC). |

## Streamlit app

### Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

### App features (assignment checklist)

- CSV upload for test data (or use bundled `test_data.csv`)
- Model selection dropdown (all 5 trained models)
- Display of Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix + classification report

### Live Streamlit link

> Replace this line with your Streamlit Community Cloud URL after deploy:
>
> `https://share.streamlit.io/<your-app>`

## How to reproduce training

1. Open `Customer_Churn_Prediction.ipynb` and **Run All**, **or**
2. Run: `python train_and_export.py` (uses the same pipeline / seed)

This regenerates `model/*.joblib` and `test_data.csv`.
