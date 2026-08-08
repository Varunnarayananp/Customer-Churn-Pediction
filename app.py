"""
Customer Churn Prediction — Streamlit app
Upload test CSV, pick a model, view metrics + confusion matrix / report.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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

st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path("model")
MODEL_OPTIONS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Gaussian Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}
TARGET_COL = "Churn Label"


def inject_style():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
  font-family: 'Public Sans', sans-serif;
}

.stApp {
  background:
    linear-gradient(90deg, #1B4332 0px, #1B4332 10px, transparent 10px),
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 27px,
      rgba(27, 67, 50, 0.035) 28px
    ),
    #F5F7F6;
  color: #14201A;
}

.block-container {
  padding-top: 1.1rem;
  padding-bottom: 2.8rem;
  max-width: 1120px;
}

#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  padding-bottom: 1.1rem;
  border-bottom: 2px solid #1B4332;
  margin-bottom: 1.35rem;
  animation: slidein 0.55s ease-out both;
}
.brand-block .eyebrow {
  font-size: 0.75rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-weight: 700;
  color: #5B7366;
  margin-bottom: 0.35rem;
}
.brand-block .name {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.6rem, 5vw, 3.8rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 0.92;
  color: #1B4332;
  margin: 0;
}
.brand-block .name .nowrap {
  white-space: nowrap;
}
.brand-block .name em {
  font-style: normal;
  color: #D97706;
}
.top-note {
  max-width: 18rem;
  text-align: right;
  font-size: 0.92rem;
  line-height: 1.45;
  color: #3E5448;
}

.controls {
  display: grid;
  grid-template-columns: 1.2fr 1fr 1fr;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
  animation: slidein 0.65s ease-out both;
}
@media (max-width: 900px) {
  .controls { grid-template-columns: 1fr; }
  .topbar { flex-direction: column; align-items: flex-start; }
  .top-note { text-align: left; max-width: none; }
}

.kicker {
  font-family: 'Syne', sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #1B4332;
  margin: 1.35rem 0 0.25rem 0;
}
.subcopy {
  color: #52685C;
  margin-bottom: 0.85rem;
  font-size: 0.95rem;
}

.score-row {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0;
  border-top: 2px solid #1B4332;
  border-bottom: 2px solid #1B4332;
  margin: 0.4rem 0 1.2rem 0;
  animation: slidein 0.7s ease-out both;
}
@media (max-width: 900px) {
  .score-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
.score-item {
  padding: 0.85rem 0.7rem;
  border-right: 1px solid #B7C7BD;
}
.score-item:last-child { border-right: none; }
.score-item .lbl {
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 700;
  color: #62786C;
}
.score-item .num {
  font-family: 'Syne', sans-serif;
  font-size: 1.7rem;
  font-weight: 700;
  color: #14201A;
  margin-top: 0.15rem;
  letter-spacing: -0.03em;
}

.meta-line {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1.4rem;
  font-size: 0.9rem;
  color: #3E5448;
  margin-bottom: 0.9rem;
  animation: slidein 0.5s ease-out both;
}
.meta-line b { color: #1B4332; }

.split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-bottom: 0.5rem;
}
@media (max-width: 900px) {
  .split { grid-template-columns: 1fr; }
}
.block-head {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: #1B4332;
  border-bottom: 1px solid #1B4332;
  padding-bottom: 0.35rem;
  margin-bottom: 0.75rem;
}
.report-box {
  font-family: ui-monospace, 'Cascadia Mono', monospace;
  font-size: 0.8rem;
  white-space: pre;
  overflow-x: auto;
  background: #14201A;
  color: #E7F0EA;
  padding: 1rem;
  border-top: 4px solid #D97706;
  line-height: 1.4;
}

div[data-testid="stDataFrame"] {
  border: 1px solid #1B4332;
  background: rgba(255,255,255,0.72);
}

div[data-testid="stFileUploader"] {
  border: 2px dashed rgba(27, 67, 50, 0.65);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  padding: 0.4rem;
}
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] input[type="file"] {
  background-color: #D97706 !important;
  color: #ffffff !important;
  font-weight: 700;
  border-radius: 999px !important;
  padding: 0.75rem 1rem !important;
  border: none !important;
}
div[data-testid="stFileUploader"] button:hover {
  background-color: #b45309 !important;
}

@keyframes slidein {
  from { opacity: 0; transform: translateX(-12px); }
  to { opacity: 1; transform: translateX(0); }
}
</style>
        """,
        unsafe_allow_html=True,
    )


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

    return data.drop(columns=[TARGET_COL]), y.astype(int)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def score_row_html(metrics: dict) -> str:
    items = [
        ("Accuracy", "Accuracy"),
        ("AUC", "AUC"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1", "F1"),
        ("MCC", "MCC"),
    ]
    html = ['<div class="score-row">']
    for label, key in items:
        html.append(
            f'<div class="score-item"><div class="lbl">{label}</div>'
            f'<div class="num">{metrics[key]:.4f}</div></div>'
        )
    html.append("</div>")
    return "".join(html)


def plot_confusion(cm: np.ndarray, model_name: str):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.patch.set_facecolor("#F5F7F6")
    ax.set_facecolor("#F5F7F6")

    for i in range(2):
        for j in range(2):
            correct = i == j
            face = "#1B4332" if correct else "#D97706"
            alpha = 0.18 + 0.62 * (cm[i, j] / max(float(cm.max()), 1.0))
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#14201A",
                    linewidth=1.4,
                    alpha=alpha,
                )
            )
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color="#14201A",
            )

    ax.set_xticks([0, 1], ["No Churn", "Churn"])
    ax.set_yticks([0, 1], ["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_title(model_name, fontsize=12, fontweight="bold", color="#1B4332", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


inject_style()

st.markdown(
    """
<div class="topbar">
  <div class="brand-block">
    <div class="eyebrow"></div>
    <p class="name">Customer Churn Predictor</p>
  </div>
  <div class="top-note">
    Pick a model, load test customers, and read who is likely to leave — with full metrics and matrix.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.3, 1.1, 1.1])
with c1:
    model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
with c2:
    uploaded = st.file_uploader("Test CSV", type=["csv"])
with c3:
    use_bundled = st.checkbox("Use bundled test data", value=uploaded is None)
    st.caption("Keep checked if you are not uploading a file.")

model_path = MODEL_OPTIONS[model_name]
if not model_path.exists():
    st.error(f"Model file not found: `{model_path}`. Run the training notebook first.")
    st.stop()

model = load_model(str(model_path))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_name = uploaded.name
elif use_bundled:
    bundled = next(
        (p for p in [Path("test_data.csv"), Path("test_data_new.csv")] if p.exists()),
        None,
    )
    if bundled is None:
        st.warning("No bundled test CSV found. Upload a CSV to evaluate a model.")
        st.stop()
    df = pd.read_csv(bundled)
    source_name = bundled.name
else:
    st.warning("Upload a CSV (or enable bundled test data) to evaluate a model.")
    st.stop()

X, y_true = prepare_xy(df)

try:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(f"Prediction failed. Check that columns match training features.\n\n{exc}")
    st.stop()

metrics = compute_metrics(y_true, y_pred, y_prob)

st.markdown(
    f"""
<div class="meta-line">
  <div><b>Model</b> {model_name}</div>
  <div><b>File</b> {source_name}</div>
  <div><b>Rows</b> {len(df):,}</div>
  <div><b>Actual churn</b> {float(y_true.mean()):.1%}</div>
  <div><b>Predicted churn</b> {float(y_pred.mean()):.1%}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="kicker">Scoreboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subcopy">Six assignment metrics for <b>{model_name}</b></div>',
    unsafe_allow_html=True,
)
st.markdown(score_row_html(metrics), unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    st.markdown('<div class="block-head">Confusion matrix</div>', unsafe_allow_html=True)
    fig = plot_confusion(confusion_matrix(y_true, y_pred), model_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.markdown('<div class="block-head">Classification report</div>', unsafe_allow_html=True)
    report = classification_report(
        y_true, y_pred, target_names=["No Churn", "Churn"], digits=3
    )
    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

st.markdown('<div class="kicker">Prediction table</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subcopy">Sample customers with predicted label and probability</div>',
    unsafe_allow_html=True,
)

preview = df.copy()
preview["Predicted Churn"] = pd.Series(y_pred).map({1: "Yes", 0: "No"}).values
preview["Churn Probability"] = y_prob.round(4)
lead_cols = [
    c
    for c in [
        "Gender",
        "Contract",
        "Tenure in Months",
        "Monthly Charge",
        "Satisfaction Score",
        TARGET_COL,
        "Predicted Churn",
        "Churn Probability",
    ]
    if c in preview.columns
]
other_cols = [c for c in preview.columns if c not in lead_cols]
st.dataframe(
    preview[lead_cols + other_cols].head(20),
    use_container_width=True,
    height=340,
)

with st.expander("Show raw uploaded rows"):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{len(df)} rows · {len(df.columns)} columns")
