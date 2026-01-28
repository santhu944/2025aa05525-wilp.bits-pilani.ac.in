import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =========================
# Load Dataset
# =========================
DATA_PATH = "D:\\2025aa05525-wilp.bits-pilani.ac.in\\Machine Learning\\Assignment2\\data\\Credit_Card_Default.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# =========================
# Split Features & Target
# =========================
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization (important for LR & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")


# =========================
# Evaluation Function
# =========================
def evaluate_model(model_name, model, X_test, y_test, y_pred, y_proba):
    print(f"\n========== {model_name} ==========")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    # AUC for multiclass
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")

    print(f"Accuracy  : {acc:.4f}")
    print(f"AUC Score : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC Score : {mcc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    }


results = {}

# =========================
# 1. Logistic Regression
# =========================
lr = LogisticRegression(max_iter=2000, multi_class="multinomial")
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)

results["Logistic Regression"] = evaluate_model(
    "Logistic Regression", lr, X_test_scaled, y_test, y_pred, y_proba
)

joblib.dump(lr, "logistic.pkl")


# =========================
# 2. Decision Tree
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)

results["Decision Tree"] = evaluate_model(
    "Decision Tree", dt, X_test, y_test, y_pred, y_proba
)

joblib.dump(dt, "decision_tree.pkl")


# =========================
# 3. KNN
# =========================
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
y_proba = knn.predict_proba(X_test_scaled)

results["KNN"] = evaluate_model(
    "KNN", knn, X_test_scaled, y_test, y_pred, y_proba
)

joblib.dump(knn, "knn.pkl")


# =========================
# 4. Naive Bayes
# =========================
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_proba = nb.predict_proba(X_test)

results["Naive Bayes"] = evaluate_model(
    "Naive Bayes", nb, X_test, y_test, y_pred, y_proba
)

joblib.dump(nb, "naive_bayes.pkl")


# =========================
# 5. Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

results["Random Forest"] = evaluate_model(
    "Random Forest", rf, X_test, y_test, y_pred, y_proba
)

joblib.dump(rf, "random_forest.pkl")


# =========================
# 6. XGBoost
# =========================
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)

results["XGBoost"] = evaluate_model(
    "XGBoost", xgb, X_test, y_test, y_pred, y_proba
)

joblib.dump(xgb, "xgboost.pkl")


# =========================
# Final Summary Table
# =========================
print("\n\n================ FINAL COMPARISON TABLE ================\n")
summary = pd.DataFrame(results).T
print(summary)

summary.to_csv("model_comparison_metrics.csv", index=True)

print("\nAll models trained and saved successfully!")
print("Saved files:")
print("""
logistic.pkl
decision_tree.pkl
knn.pkl
naive_bayes.pkl
random_forest.pkl
xgboost.pkl
scaler.pkl
model_comparison_metrics.csv
""")
