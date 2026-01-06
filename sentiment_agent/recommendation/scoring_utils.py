"""
scoring_utils.py
Corrected sentiment scoring to accept numeric compound sentiment.
"""

from typing import Dict, Any, Optional

# ---------------------------
# KPI threshold scoring (0–50)
# ---------------------------

def _score_growth_threshold(value: Optional[float]) -> int:
    if value is None:
        return 0
    try:
        v = float(value)
    except Exception:
        return 0
    if v > 0.10:
        return 5
    if v > 0.05:
        return 4
    if v > 0.02:
        return 3
    if v > 0.0:
        return 2
    return 0


def _score_margin_threshold(value: Optional[float]) -> int:
    if value is None:
        return 0
    try:
        v = float(value)
    except Exception:
        return 0
    if v > 0.30:
        return 5
    if v > 0.20:
        return 4
    if v > 0.10:
        return 3
    if v > 0.05:
        return 2
    return 0


def _score_roe(value: Optional[float]) -> int:
    if value is None:
        return 0
    try:
        v = float(value)
    except Exception:
        return 0
    if v > 0.25:
        return 5
    if v > 0.18:
        return 4
    if v > 0.12:
        return 3
    if v > 0.05:
        return 2
    return 0


def _score_debt_to_equity(value: Optional[float]) -> int:
    if value is None:
        return 0
    try:
        v = float(value)
    except Exception:
        return 0
    if v < 0.2:
        return 5
    if v < 0.5:
        return 4
    if v < 1.0:
        return 3
    if v < 2.0:
        return 2
    return 0


def _score_fcf_trend(value: Optional[str]) -> int:
    if value is None:
        return 0
    v = str(value).strip().lower()
    if v in ("positive", "improving", "up"):
        return 5
    if v in ("neutral", "flat", "stable"):
        return 3
    if v in ("negative", "declining", "down"):
        return 0
    return 0


def score_kpis(kpi: Dict[str, Any]) -> float:
    """
    Returns KPI score in range [0, 50]
    """
    scores = [
        _score_growth_threshold(kpi.get("revenue_growth_yoy")),
        _score_growth_threshold(kpi.get("net_income_growth_yoy")),
        _score_margin_threshold(kpi.get("gross_margin_pct")),
        _score_margin_threshold(kpi.get("operating_margin_pct")),
        _score_roe(kpi.get("roe_pct")),
        _score_debt_to_equity(kpi.get("debt_to_equity")),
        _score_fcf_trend(kpi.get("fcf_trend")),
    ]

    avg = sum(scores) / len(scores) if scores else 0.0
    return (avg / 5.0) * 50.0


# ---------------------------
# Sentiment scoring (0–20)
# ---------------------------

def score_sentiment(compound: float) -> float:
    """
    Converts compound sentiment score [-1, +1]
    into sentiment contribution [0, 20].

    Neutral (0.0) → 10
    Positive (+1.0) → 20
    Negative (-1.0) → 0
    """
    if compound is None:
        raise ValueError("compound sentiment must not be None")

    try:
        c = float(compound)
    except Exception:
        raise ValueError("compound sentiment must be numeric")

    # Clamp for safety
    c = max(-1.0, min(1.0, c))

    # Linear mapping
    return (c + 1.0) * 10.0


# ---------------------------
# Peer scoring (0–30)
# ---------------------------

def score_peers(peer: Dict[str, Any]) -> float:
    valuation = float(peer.get("valuation_rank", 0.0) or 0.0)
    profitability = float(peer.get("profitability_rank", 0.0) or 0.0)
    growth = float(peer.get("growth_rank", 0.0) or 0.0)

    combined = 0.4 * valuation + 0.3 * profitability + 0.3 * growth
    return max(0.0, min(combined * 30.0, 30.0))


# ---------------------------
# Risk penalty
# ---------------------------

def risk_penalty(risk_level: str) -> int:
    rl = (risk_level or "").lower()
    if rl == "high":
        return -15
    if rl == "medium":
        return -5
    return 0
