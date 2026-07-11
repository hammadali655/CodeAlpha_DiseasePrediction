"""
Flask app for the Disease Prediction model (breast cancer diagnostic data).
Loads the pre-trained model (run train_model.py first) and serves
a form-based prediction UI using the 10 core diagnostic measurements.
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = None
scaler = None
feature_names = None
feature_ranges = None

# Human-friendly labels for each "mean ..." feature from the dataset
FIELD_LABELS = {
    "mean radius": "Mean radius",
    "mean texture": "Mean texture",
    "mean perimeter": "Mean perimeter",
    "mean area": "Mean area",
    "mean smoothness": "Mean smoothness",
    "mean compactness": "Mean compactness",
    "mean concavity": "Mean concavity",
    "mean concave points": "Mean concave points",
    "mean symmetry": "Mean symmetry",
    "mean fractal dimension": "Mean fractal dimension",
}


def load_artifacts():
    global model, scaler, feature_names, feature_ranges
    required = ["model.pkl", "scaler.pkl", "feature_names.pkl", "feature_ranges.pkl"]
    if not all(os.path.exists(f) for f in required):
        raise FileNotFoundError(
            "Model files not found. Run 'python train_model.py' first."
        )
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    feature_ranges = joblib.load("feature_ranges.pkl")


@app.route("/")
def index():
    fields = [
        {
            "key": name,
            "id": name.replace(" ", "_"),
            "label": FIELD_LABELS.get(name, name),
            "default": feature_ranges[name]["mean"],
            "min": feature_ranges[name]["min"],
            "max": feature_ranges[name]["max"],
        }
        for name in feature_names
    ]
    return render_template("index.html", fields=fields)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        values = []
        for name in feature_names:
            key = name.replace(" ", "_")
            values.append(float(data[key]))

        features = np.array([values])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        result = {
            "disease_present": bool(prediction),
            "confidence": round(float(max(probability)) * 100, 1),
            "disease_probability": round(float(probability[1]) * 100, 1),
            "healthy_probability": round(float(probability[0]) * 100, 1),
        }
        return jsonify(result)

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


load_artifacts()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
