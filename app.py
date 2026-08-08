"""
Customer Churn Prediction — Streamlit app
Upload test CSV, pick a model, view metrics + confusion matrix / report.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

MODEL_DIR = Path("model")
MODEL_OPTIONS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Gaussian Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}
TARGET_COL = "Churn Label"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def prepare_xy(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        st.error(f"Uploaded CSV must include a '{TARGET_COL}' column (Yes/No).")
        st.stop()

    data = df.copy()
    for col in ["Offer", "Internet Type"]:
        if col in data.columns:
            data[col] = data[col].fillna("None")

    y = data[TARGET_COL].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    if y.isnull().any():
        st.error(f"'{TARGET_COL}' must be Yes/No (or 1/0).")
        st.stop()

    X = data.drop(columns=[TARGET_COL])
    return X, y.astype(int)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("Customer Churn Prediction")
st.write(
    "Upload your **test CSV**, choose a trained model, and view evaluation metrics, "
    "confusion matrix, and classification report."
)

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Select model", list(MODEL_OPTIONS.keys()))
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    use_bundled = st.checkbox("Use bundled test_data.csv", value=True if uploaded is None else False)

model_path = MODEL_OPTIONS[model_name]
if not model_path.exists():
    st.error(f"Model file not found: {model_path}. Run the training notebook first.")
    st.stop()

model = load_model(str(model_path))

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_bundled and Path("test_data.csv").exists():
    df = pd.read_csv("test_data.csv")
    st.info("Using bundled `test_data.csv`.")
else:
    st.warning("Upload a CSV (or enable bundled test_data.csv) to evaluate a model.")
    st.stop()

st.subheader("Uploaded data preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

X, y_true = prepare_xy(df)

try:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(f"Prediction failed. Check that columns match training features.\n\n{exc}")
    st.stop()

metrics = compute_metrics(y_true, y_pred, y_prob)

st.subheader(f"Evaluation metrics — {model_name}")
m1, m2, m3 = st.columns(3)
m4, m5, m6 = st.columns(3)
m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
m2.metric("AUC", f"{metrics['AUC']:.4f}")
m3.metric("Precision", f"{metrics['Precision']:.4f}")
m4.metric("Recall", f"{metrics['Recall']:.4f}")
m5.metric("F1 Score", f"{metrics['F1']:.4f}")
m6.metric("MCC", f"{metrics['MCC']:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(model_name)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=False
    )
    st.text(report)

st.subheader("Prediction sample")
preview = df.copy()
preview["Predicted Churn"] = pd.Series(y_pred).map({1: "Yes", 0: "No"}).values
preview["Churn Probability"] = y_prob.round(4)
st.dataframe(preview.head(20), use_container_width=True)
