"""
FastAPI app for sentiment prediction.
"""
import os
import re
import sys
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Make text_preprocess importable (required by the pickled model pipeline)
sys.path.insert(0, os.path.dirname(__file__))

model = None


def get_model():
    global model
    if model is None:
        model = joblib.load(
            os.path.join(os.path.dirname(__file__), "..", "classification_logreg.joblib")
        )
    return model

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    sentiment: str
    confidence: Optional[float] = None


# ---------- Helpers ----------

def format_prediction(raw_pred) -> Tuple[str, str]:
    pred = str(raw_pred).strip().lower()
    if pred in {"positive", "pos", "1", "good", "true"}:
        return "Good review", "good"
    if pred in {"negative", "neg", "0", "bad", "false"}:
        return "Bad review", "bad"
    return f"Prediction: {raw_pred}", "neutral"


def negation_override(text: str) -> Tuple[Optional[str], Optional[str]]:
    lowered = text.lower()
    positive_pattern = r"\bnot\s+(?:\w+\s+){0,2}(bad|terrible|awful|horrible|poor|worst|disappointing|weak)\b"
    negative_pattern = r"\bnot\s+(?:\w+\s+){0,2}(good|great|amazing|excellent|awesome|nice|recommend|worth|enjoyable|fun|interesting|engaging)\b"
    if re.search(positive_pattern, lowered):
        return "Good review", "good"
    if re.search(negative_pattern, lowered):
        return "Bad review", "bad"
    return None, None


# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter a review.")

    override_label, override_sentiment = negation_override(text)

    m = get_model()
    confidence = None
    if hasattr(m, "predict_proba"):
        try:
            proba = m.predict_proba([text])[0]
            confidence = float(max(proba) * 100)
        except Exception:
            pass

    if override_label:
        return PredictResponse(
            label=override_label,
            sentiment=override_sentiment,
            confidence=confidence,
        )

    raw_pred = m.predict([text])[0]
    label, sentiment = format_prediction(raw_pred)
    return PredictResponse(
        label=label,
        sentiment=sentiment,
        confidence=confidence,
    )


# Vercel ASGI handler
from mangum import Mangum  # noqa: E402
handler = Mangum(app)
