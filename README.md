# CodeAlpha_DiseasePrediction

Predicting disease presence from patient medical data, built as part of the **CodeAlpha Machine Learning Internship**.

## 📌 Objective
Predict the possibility of disease (breast cancer) from structured patient diagnostic data using classification techniques.

## 🧠 Approach
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset — a real, well-established medical dataset (569 patients, 30 diagnostic features)
- **Models trained:** Logistic Regression, Random Forest, SVM (RBF kernel), XGBoost
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Feature Importance

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 96.49% | 0.975 | 0.929 | 0.951 | **0.9960** |
| Random Forest | 97.37% | 1.000 | 0.929 | 0.963 | 0.9944 |
| SVM (RBF) | 97.37% | 1.000 | 0.929 | 0.963 | 0.9947 |
| XGBoost | 97.37% | 1.000 | 0.929 | 0.963 | 0.9934 |

All four models perform strongly, with **Logistic Regression achieving the best ROC-AUC (0.9960)** and the tree/kernel-based models achieving the highest raw accuracy (97.37%).

![Confusion Matrix](confusion_matrix.png)

## 🗂 Dataset
Uses the **Breast Cancer Wisconsin dataset**, built directly into scikit-learn — no download required. Instructions for swapping in a Heart Disease or Diabetes dataset from Kaggle/UCI instead are included as comments in `disease_prediction.py`.

## 🛠 Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn

## 🚀 How to Run
```bash
pip install -r requirements.txt
python disease_prediction.py
```
Outputs classification reports for all 4 models, a confusion matrix for the best model, and a feature importance chart.

## 📁 Files
- `disease_prediction.py` — full training & evaluation pipeline
- `requirements.txt` — dependencies
- `confusion_matrix.png` — confusion matrix for best model (generated on run)
- `feature_importance.png` — top 10 predictive features (generated on run)

---
*Part of the CodeAlpha Machine Learning Internship — [www.codealpha.tech](https://www.codealpha.tech)*
