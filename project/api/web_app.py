"""
Flask app that serves a modern web UI for sentiment prediction.
"""
import os
import re
from typing import Tuple
from flask import Flask, jsonify, render_template, request
import joblib

MODEL_FILENAME = "classification_logreg.joblib"
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", MODEL_FILENAME)
)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "web", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "web", "static"),
)


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Run train_baselines.py first."
        )
    return joblib.load(path)


MODEL = load_model(MODEL_PATH)


def format_prediction(raw_pred) -> Tuple[str, str]:
    pred = str(raw_pred).strip().lower()
    if pred in {"positive", "pos", "1", "good", "true"}:
        return "Good review", "good"
    if pred in {"negative", "neg", "0", "bad", "false"}:
        return "Bad review", "bad"
    return f"Prediction: {raw_pred}", "neutral"


def negation_override(text: str) -> Tuple[str, str] | Tuple[None, None]:
    lowered = text.lower()
    positive_pattern = r"\bnot\s+(?:\w+\s+){0,2}(bad|terrible|awful|horrible|poor|worst|disappointing|weak)\b"
    negative_pattern = r"\bnot\s+(?:\w+\s+){0,2}(good|great|amazing|excellent|awesome|nice|recommend|worth|enjoyable|fun|interesting|engaging)\b"

    if re.search(positive_pattern, lowered):
        return "Good review", "good"
    if re.search(negative_pattern, lowered):
        return "Bad review", "bad"
    return None, None


def get_confidence(text: str) -> float | None:
    if not hasattr(MODEL, "predict_proba"):
        return None
    try:
        proba = MODEL.predict_proba([text])[0]
        return float(max(proba) * 100)
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a review."}), 400

    try:
        override_label, override_sentiment = negation_override(text)
        if override_label:
            confidence = get_confidence(text)
            return jsonify(
                {
                    "label": override_label,
                    "sentiment": override_sentiment,
                    "confidence": confidence,
                }
            )

        raw_pred = MODEL.predict([text])[0]
        label, sentiment = format_prediction(raw_pred)

        confidence = get_confidence(text)

        return jsonify(
            {
                "label": label,
                "sentiment": sentiment,
                "confidence": confidence,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
