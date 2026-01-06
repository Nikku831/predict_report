"""
recommendation_engine.py
Corrected sentiment handling and pathing.
"""

from typing import Dict, Any, Optional
import json
import os
import logging
from transformers import pipeline
import torch

from .scoring_utils import (
    score_kpis,
    score_sentiment,
    score_peers,
    risk_penalty
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# ✅ FIXED PATH LOGIC (Solves FileNotFoundError)
# --------------------------------------------------
# Get the directory where THIS file (recommendation_engine.py) lives
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to 'sentiment_agent', then into 'config'
# Structure assumed: sentiment_agent/recommendation/../config -> sentiment_agent/config
CONFIG_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'config')

def _load_json(filename: str) -> dict:
    """
    Loads a JSON file from the verified CONFIG_DIR.
    """
    path = os.path.join(CONFIG_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback for debugging if paths are slightly different in your setup
        logger.error(f"Config file not found at: {path}")
        raise

# Load configs using just the filename (path is handled automatically now)
RATING_THRESHOLDS = _load_json("rating_thresholds.json")
RISK_PENALTIES = _load_json("risk_penalities.json")
WEIGHTAGES = _load_json("weightages.json")

_DEVICE = 0 if torch.cuda.is_available() else -1
_llm = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    device=_DEVICE
)

# --------------------------------------------------
# ✅ FIXED RISK ASSESSMENT
# --------------------------------------------------
def _assess_risk_from_payload(payload: Dict[str, Any]) -> str:
    kpis = payload.get("kpis", {})

    try:
        dte = float(kpis.get("debt_to_equity", 0))
    except Exception:
        dte = 0.0

    if dte > 1.0:
        return "high"
    if dte > 0.5:
        return "medium"

    try:
        # Check sentiment_score safely
        compound = payload.get("sentiment_score")
        if compound is None:
             # Fallback if sentiment is missing or 0.0
             logger.warning("Sentiment score missing in payload, defaulting to 0.0 for risk check.")
             compound = 0.0
        else:
             compound = float(compound)
    except Exception:
        compound = 0.0

    if compound < 0.2:
        return "high"
    if compound < 0.4:
        return "medium"
    return "low"

def _generate_local_explanation(prompt: str) -> Optional[str]:
    try:
        output = _llm(
            prompt,
            max_new_tokens=80,
            do_sample=False,
            num_beams=2,
            early_stopping=True
        )
        text = output[0]["generated_text"].strip()
        text = text.replace("-LRB-", "(").replace("-RRB-", ")").replace("  ", " ")
        return text
    except Exception as e:
        logger.exception("Local LLM explanation failed: %s", e)
        return None

def _build_prompt(payload: Dict[str, Any], scores: Dict[str, float]) -> str:
    rating = payload.get("rating", "HOLD")
    risk = payload.get("risk_level", "medium")
    return (
        "Rewrite the following financial summary in professional language.\n\n"
        f"Summary:\n"
        f"The KPI score is {scores['kpi']}. "
        f"The sentiment score is {scores['sentiment']}. "
        f"The peer comparison score is {scores['peer']}. "
        f"The overall risk level is {risk}. "
        f"The final recommendation is {rating}.\n\n"
        "Rewrite this as a short 3–4 sentence investment rationale."
    )

# --------------------------------------------------
# ✅ FIXED RECOMMENDATION PIPELINE
# --------------------------------------------------
def generate_recommendation(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary")

    kpis = payload.get("kpis", {})
    peer_data = payload.get("peer_data", {})

    sentiment_score_input = payload.get("sentiment_score")
    
    # Relaxed check: default to 0.0 if missing to prevent crashing
    if sentiment_score_input is None:
        logger.warning("sentiment_score missing, defaulting to 0.0")
        sentiment_score_input = 0.0

    kpi_score = score_kpis(kpis)
    sentiment_score = score_sentiment(sentiment_score_input)
    peer_score = score_peers(peer_data)

    risk_level = _assess_risk_from_payload(payload)
    penalty = risk_penalty(risk_level)

    final_score = kpi_score + sentiment_score + peer_score + penalty
    final_score = max(0.0, min(final_score, 100.0))

    buy_min = RATING_THRESHOLDS.get("buy", 70) # Added defaults for safety
    hold_min = RATING_THRESHOLDS.get("hold", 40)

    if final_score >= buy_min:
        rating = "BUY"
    elif final_score >= hold_min:
        rating = "HOLD"
    else:
        rating = "SELL"

    confidence = round(final_score / 100.0, 2)

    scores = {
        "kpi": round(kpi_score, 2),
        "sentiment": round(sentiment_score, 2),
        "peer": round(peer_score, 2),
        "penalty": round(penalty, 2),
        "final": round(final_score, 2),
    }

    payload["rating"] = rating
    payload["risk_level"] = risk_level

    prompt = _build_prompt(payload, scores)
    explanation = _generate_local_explanation(prompt)

    if not explanation:
        explanation = (
            f"The company shows moderate financial strength with a KPI score of "
            f"{scores['kpi']} and sentiment score of {scores['sentiment']}. "
            f"The assessed risk level is {risk_level}. "
            f"Based on combined indicators, the system recommends a {rating} stance."
        )

    return {
        "rating": rating,
        "final_score": round(final_score, 2),
        "confidence": confidence,
        "reasoning": {
            "analysis": explanation,
            "numeric_scores": scores,
            "risk_level": risk_level,
        },
    }