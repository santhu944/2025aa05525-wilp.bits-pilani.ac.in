import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Page Configuration
# ================================
st.set_page_config(page_title="Credit Card Risk Prediction", layout="wide")

st.title("💳 Credit Card Risk Level Prediction System")
st.write("""
This Streamlit app evaluates multiple machine learning models for predicting 
Credit Card Risk Levels:
- 0 → Low Risk  
- 1 → Medium Risk  
- 2 → High Risk  
""")

# ================================
# Load Models
# ================================
MODEL_PATHS = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
    "XGBoost (Ensemble)": "model/xgboost.pkl"
}

SCALER_PATH = "model/scaler.pkl"

# ================================
# Sidebar
# ================================
st.sidebar.header("Model Configuration")
model_name = st.sidebar.selectbox(
    "Select a Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

evaluate_btn = st.sidebar.button("Evaluate Model")

# ================================
# Helper Functions
# ================================
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

def load_scaler():
    return joblib.load(SCALER_PATH)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)

    # AUC for multiclass
    y_bin = label_binarize(y, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")

    cm = confusion_matrix(y, y_pred)

    return acc, auc, precision, recall, f1, mcc, cm

# ================================
# Main Logic
# ================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Dataset Preview")
    st.dataframe(df.head())

    if "risk_level" not in df.columns:
        st.error("Uploaded CSV must contain 'risk_level' column as target.")
    else:
        X = df.drop("risk_level", axis=1)
        y = df["risk_level"]

        if evaluate_btn:
            model = load_model(model_name)

            # Scaling only for models that need it
            if model_name in ["Logistic Regression", "KNN"]:
                scaler = load_scaler()
                X_processed = scaler.transform(X)
            else:
                X_processed = X

            acc, auc, precision, recall, f1, mcc, cm = evaluate_model(
                model, X_processed, y
            )

            st.subheader(f"📈 Evaluation Metrics for {model_name}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col1.metric("AUC Score", f"{auc:.4f}")

            col2.metric("Precision", f"{precision:.4f}")
            col2.metric("Recall", f"{recall:.4f}")

            col3.metric("F1 Score", f"{f1:.4f}")
            col3.metric("MCC Score", f"{mcc:.4f}")

            st.subheader("🧮 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file from the sidebar to begin evaluation.")

# ================================
# Footer
# ================================
st.markdown("""
---
Developed for **ML Assignment – 2 (BITS Pilani)**  
M.Tech (AIML / DSE) | Work Integrated Learning Programmes
""")
