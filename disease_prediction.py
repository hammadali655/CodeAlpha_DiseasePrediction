"""
CodeAlpha - Machine Learning Internship
TASK 4: Disease Prediction from Medical Data
--------------------------------------------------------------
Objective : Predict the possibility of disease from patient data.
Dataset   : Breast Cancer Wisconsin (Diagnostic) Dataset
            -> This is a REAL dataset, built directly into
               scikit-learn, so it needs NO internet download.
Algorithms: Logistic Regression, Random Forest, SVM
            (XGBoost included automatically if installed)
--------------------------------------------------------------
WANT A DIFFERENT DISEASE DATASET INSTEAD? (Heart Disease / Diabetes)
Download from UCI / Kaggle and place the CSV in this folder, e.g.:
  - Heart Disease: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
  - Diabetes:      https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Then set DATA_SOURCE = "csv" below and point CSV_PATH / TARGET_COL
to match the file you downloaded.
--------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

RANDOM_STATE = 42
DATA_SOURCE = "sklearn"   # "sklearn" (built-in, real data) or "csv"
CSV_PATH = "disease_data.csv"
TARGET_COL = "target"


def load_data():
    if DATA_SOURCE == "sklearn":
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        # sklearn encodes 0 = malignant, 1 = benign -> flip so 1 = disease present
        df["target"] = 1 - df["target"]
        return df, "target"
    else:
        df = pd.read_csv(CSV_PATH)
        return df, TARGET_COL


def main():
    df, target_col = load_data()
    print("Dataset shape:", df.shape)
    print("\nFirst rows:")
    print(df.head())
    print(f"\nClass balance ({target_col}):")
    print(df[target_col].value_counts(normalize=True))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE
        )
    except ImportError:
        print("\n(xgboost not installed - skipping XGBoost model. "
              "Install it with `pip install xgboost` to include it.)")

    results = {}
    best_model_name, best_model_obj, best_auc = None, None, -1

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

        print(f"\n===== {name} =====")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
        print(f"ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc, best_model_name, best_model_obj = auc, name, model

    print("\n===== Summary (all models) =====")
    summary_df = pd.DataFrame(results).T
    print(summary_df.round(4))
    print(f"\nBest model: {best_model_name} (ROC-AUC = {best_auc:.4f})")

    # Confusion matrix plot for best model
    y_pred_best = best_model_obj.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to confusion_matrix.png")

    # Feature importance if available (tree-based models)
    if hasattr(best_model_obj, "feature_importances_"):
        importances = pd.Series(best_model_obj.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(10)
        plt.figure(figsize=(7, 5))
        top_features.sort_values().plot(kind="barh", color="mediumpurple")
        plt.title(f"Top 10 Feature Importances - {best_model_name}")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        print("Saved feature importance chart to feature_importance.png")


if __name__ == "__main__":
    main()
