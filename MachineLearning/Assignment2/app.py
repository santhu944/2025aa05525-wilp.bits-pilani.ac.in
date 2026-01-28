import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Path Handling (IMPORTANT FIX)
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATHS = {
    "Decision Tree": os.path.join(MODEL_DIR, "decision_tree.pkl"),
    "KNN": os.path.join(MODEL_DIR, "knn.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "xgboost.pkl"),
}

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# ================================
# Helper Functions
# ================================
def load_model(model_name):
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found: {SCALER_PATH}")
        st.stop()
    return joblib.load(SCALER_PATH)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="ML Assignment 2 - Model Evaluation", layout="wide")
st.title("📊 Machine Learning Model Evaluation Dashboard")

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

evaluate_btn = st.button("Evaluate Model")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset")
    st.dataframe(df.head())

    if "risk_level" not in df.columns:
        st.error("Your dataset must contain a 'risk_level' column as target.")
        st.stop()

    X = df.drop("risk_level", axis=1)
    y = df["risk_level"]

    if evaluate_btn:
        model = load_model(model_name)

        # Scaling only for specific models
        if model_name in ["Logistic Regression", "KNN"]:
            scaler = load_scaler()
            X = scaler.transform(X)

        # Predictions
        y_pred = model.predict(X)

        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="weighted")
        rec = recall_score(y, y_pred, average="weighted")
        f1 = f1_score(y, y_pred, average="weighted")
        mcc = matthews_corrcoef(y, y_pred)

        # AUC (Multi-class safe)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
            try:
                auc = roc_auc_score(
                    y,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                )
            except ValueError:
                auc = 0.0
        else:
            auc = 0.0

        # Display Metrics
        st.subheader(f"📌 Evaluation Metrics for {model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("F1 Score", f"{f1:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("AUC Score", f"{auc:.4f}")
        col5.metric("Recall", f"{rec:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        class_names = sorted(y.unique())
        plot_confusion_matrix(cm, class_names)
