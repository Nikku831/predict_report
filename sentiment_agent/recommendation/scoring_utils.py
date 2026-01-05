"""
scoring_utils.py
Original file content restored.
"""

from typing import Dict, Any

# ---------------------------
# KPI threshold scoring (0-5 per KPI)
# ---------------------------

def _score_growth_threshold(value: float) -> int:
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

def _score_margin_threshold(value: float) -> int:
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

def _score_roe(value: float) -> int:
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

def _score_debt_to_equity(value: float) -> int:
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

def _score_fcf_trend(value: str) -> int:
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
    per_kpi_scores = []
    per_kpi_scores.append(_score_growth_threshold(kpi.get("revenue_growth")))
    per_kpi_scores.append(_score_growth_threshold(kpi.get("net_income_growth")))
    per_kpi_scores.append(_score_growth_threshold(kpi.get("eps_growth")))
    per_kpi_scores.append(_score_margin_threshold(kpi.get("gross_margin")))
    per_kpi_scores.append(_score_margin_threshold(kpi.get("operating_margin")))
    per_kpi_scores.append(_score_roe(kpi.get("roe")))
    per_kpi_scores.append(_score_debt_to_equity(kpi.get("debt_to_equity")))
    per_kpi_scores.append(_score_fcf_trend(kpi.get("fcf_trend")))

    avg = sum(per_kpi_scores) / len(per_kpi_scores) if per_kpi_scores else 0.0
    return (avg / 5.0) * 50.0

# ---------------------------
# Sentiment scoring (0-20)
# ---------------------------

def score_sentiment(sentiment: Dict[str, Any]) -> float:
    score = 0
    overall = sentiment.get("overall_score")
    try:
        o = float(overall) if overall is not None else 0.0
    except Exception:
        o = 0.0

    if o > 0.6: score += 8
    elif o > 0.3: score += 5
    else: score += 1

    tone = str(sentiment.get("tone", "")).lower()
    if tone == "confident": score += 6
    elif tone == "neutral": score += 3
    else: score += 1

    hedging = sentiment.get("hedging")
    try:
        h = int(hedging) if hedging is not None else 0
    except Exception:
        h = 0
    if h < 5: score += 3
    elif h < 10: score += 2

    um = sentiment.get("uncertainty_markers")
    try:
        um_v = int(um) if um is not None else 999
    except Exception:
        um_v = 999
    if um_v < 5: score += 3
    elif um_v < 10: score += 1

    return min(float(score), 20.0)

# ---------------------------
# Peer scoring (0-30)
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
    if rl == "high": return -15
    if rl == "medium": return -5
    return 0