# ML Assignment – 2  
**M.Tech (AIML / DSE) – Work Integrated Learning Programmes**  
**Course:** Machine Learning  

---

## a. Problem Statement  

The objective of this project is to build and evaluate multiple machine learning classification models for predicting the **Credit Card Risk Level** of customers.  

The risk is categorized into three classes:  

- **0 → Low Risk**  
- **1 → Medium Risk**  
- **2 → High Risk**  

This project demonstrates the complete end-to-end ML workflow:  
- Dataset preparation  
- Model training and evaluation  
- Comparison of multiple classifiers  
- Development of an interactive Streamlit web application  
- Deployment on Streamlit Community Cloud  

---

## b. Dataset Description  **[1 Mark]**

We use a **Synthetic Credit Card Risk Dataset** generated to mimic real-world credit card customer behavior.  
The dataset is intentionally synthetic to ensure originality and to avoid plagiarism issues.

**Dataset Properties:**
- Total Samples: **30,000**
- Total Features: **23+**
- Target Column: `risk_level` (Multiclass)

| Risk Level | Meaning |
|-----------|--------|
| 0 | Low Risk |
| 1 | Medium Risk |
| 2 | High Risk |

**Important Features:**

| Feature Name | Description |
|-------------|------------|
| LIMIT_BAL | Credit limit assigned to the customer |
| AGE | Age of customer |
| INCOME | Monthly income |
| UTILIZATION | Credit utilization ratio |
| PAY_1 … PAY_6 | Past payment history |
| BILL_AMT1 … BILL_AMT6 | Bill amounts of previous months |
| PAY_AMT1 … PAY_AMT6 | Payment amounts of previous months |
| risk_level | Target class |

Two CSV files are used:
- `Credit_Card_Default.csv` → Full dataset for training  
- `credit_test_sample.csv` → Small test dataset for Streamlit upload  

---

## c. Models Used and Evaluation Metrics  **[6 Marks]**

The following six machine learning models were implemented on the same dataset:

1. Logistic Regression (Multinomial)  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

For each model, the following metrics were calculated:

- Accuracy  
- AUC Score (One-vs-Rest for multiclass)  
- Precision (Weighted)  
- Recall (Weighted)  
- F1 Score (Weighted)  
- Matthews Correlation Coefficient (MCC)  

### 📊 Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|----------|-------|--------|-----|
| Logistic Regression | 0.6178 | 0.7359 | 0.6174 | 0.6178 | 0.6176 | 0.3309 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.6813 | 0.7863 | 0.6929 | 0.6813 | 0.6765 | 0.4337 |
| Naive Bayes | 0.4968 | 0.5906 | 0.4510 | 0.4968 | 0.4614 | 0.0686 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

> The above values are directly taken from `model_comparison_metrics.csv` generated during model training.

---

## d. Observations on Model Performance  **[3 Marks]**

| ML Model | Observation about model performance |
|--------|------------------------------------|
| Logistic Regression | Performs well as a baseline model and provides stable results. However, it struggles to capture complex non-linear patterns present in the dataset. |
| Decision Tree | Achieves perfect performance on this dataset, indicating strong learning capability. However, such perfect scores suggest possible overfitting. |
| KNN | Shows better performance than Logistic Regression and Naive Bayes. It benefits from feature scaling but can become computationally expensive for large datasets. |
| Naive Bayes | Fast and simple, but performance is limited due to its assumption of feature independence, which is unrealistic in financial datasets. |
| Random Forest (Ensemble) | Produces excellent and stable performance by combining multiple trees. It handles non-linearity and feature interactions very effectively. |
| XGBoost (Ensemble) | Shows the best overall performance. Its boosting strategy and regularization make it highly powerful for complex classification problems. |

---

## Project Structure

