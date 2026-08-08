"""Train all 5 models, export artifacts for Streamlit, print metrics."""
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("archive/telco.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("telco.csv")

df = pd.read_csv(DATA_PATH)

leakage_or_id_cols = [
    "Customer ID",
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
    "Satisfaction Score",
    "Country",
    "State",
    "Quarter",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
]
df = df.drop(columns=[c for c in leakage_or_id_cols if c in df.columns])
for col in ["Offer", "Internet Type"]:
    if col in df.columns:
        df[col] = df[col].fillna("None")

y = df["Churn Label"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn Label"])
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)
gnb_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def evaluate_model(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_te, y_pred),
        "AUC": roc_auc_score(y_te, y_prob),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "MCC": matthews_corrcoef(y_te, y_pred),
    }
    print(f"\n========== {name} ==========\n")
    for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        print(f"{k:10s}: {metrics[k]:.4f}")
    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_te, y_pred))
    print("\nClassification Report\n")
    print(classification_report(y_te, y_pred))
    return metrics


models = {
    "Logistic Regression": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    ),
    "Decision Tree": Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42),
            ),
        ]
    ),
    "KNN": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=11)),
        ]
    ),
    "Gaussian Naive Bayes": Pipeline(
        [
            ("preprocessor", gnb_preprocessor),
            ("classifier", GaussianNB()),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

file_map = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

os.makedirs("model", exist_ok=True)
all_metrics = []
for name, model in models.items():
    model.fit(X_train, y_train)
    all_metrics.append(evaluate_model(name, model, X_test, y_test))
    joblib.dump(model, f"model/{file_map[name]}")

results_df = pd.DataFrame(all_metrics)
comparison = results_df.copy()
for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
    comparison[col] = comparison[col].round(4)

print("\n===== Comparison Table =====\n")
print(comparison.to_string(index=False))
best_row = results_df.loc[results_df["F1"].idxmax()]
print(
    f"\nOverall winner (by F1): {best_row['Model']} "
    f"(F1={best_row['F1']:.4f}, AUC={best_row['AUC']:.4f}, MCC={best_row['MCC']:.4f})"
)

test_export = X_test.copy()
test_export["Churn Label"] = y_test.map({1: "Yes", 0: "No"}).values
test_export.to_csv("test_data.csv", index=False)
comparison.to_csv("model/metrics_comparison.csv", index=False)

print("\nSaved models to model/")
print("Saved test_data.csv with", len(test_export), "rows")
print("Features used:", X.shape[1])
