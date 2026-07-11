"""
Trains the Disease Prediction model (Breast Cancer Wisconsin dataset,
built into scikit-learn) and saves it for the Flask app to load.

Run this once before starting the Flask app:
    python train_model.py
"""

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target"] = 1 - df["target"]  # 1 = disease present, matches app framing

    # Use the 10 "mean" features only - keeps the web form short and readable
    # while still capturing the strongest predictive signal.
    feature_cols = [c for c in df.columns if c.startswith("mean ")]

    X = df[feature_cols]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_cols, "feature_names.pkl")

    # Save typical ranges (min/max/mean) for each feature, used to
    # pre-fill the form with sensible example values in the UI.
    ranges = {
        col: {
            "min": round(float(X[col].min()), 3),
            "max": round(float(X[col].max()), 3),
            "mean": round(float(X[col].mean()), 3),
        }
        for col in feature_cols
    }
    joblib.dump(ranges, "feature_ranges.pkl")

    print("Model trained and saved: model.pkl, scaler.pkl, feature_names.pkl, feature_ranges.pkl")
    print(f"Training accuracy: {model.score(X_scaled, y):.4f}")
    print(f"Features used: {feature_cols}")


if __name__ == "__main__":
    main()
