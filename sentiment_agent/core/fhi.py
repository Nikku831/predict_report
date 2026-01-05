from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path

# ------------------------------------------------------------
# CONFIG PATH (your updated YAML with sentiment scoring added)
# ------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "kpi_thresholds.yaml"

# ------------------------------------------------------------
# FHI WEIGHTS (Sentiment added)
# ------------------------------------------------------------
FHI_WEIGHTS = {
    "profitability": 0.30,
    "liquidity": 0.20,
    "solvency": 0.15,
    "growth": 0.20,
    "sentiment": 0.15   # NEW SENTIMENT DIMENSION
}

# ------------------------------------------------------------
# Dimension → Metrics Mapping
# ------------------------------------------------------------
FHI_DIMENSIONS = {
    "profitability": ["gross_margin_pct", "operating_margin_pct", "net_margin_pct", "roe_pct"],
    "liquidity": ["current_ratio", "quick_ratio"],
    "solvency": ["debt_to_equity"],
    "growth": ["revenue_growth_yoy", "net_income_growth_yoy"],
    
    # NEW SENTIMENT METRICS
    "sentiment": [
        "compound_sentiment",
        "tone_positive_ratio",
        "tone_negative_ratio",
        "uncertainty_index",
        "hedging_index"
    ]
}


# ------------------------------------------------------------
# Load KPI + Sentiment Scoring Thresholds
# ------------------------------------------------------------
def _load_thresholds() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


THRESHOLDS = _load_thresholds()


# ------------------------------------------------------------
# Scoring Functions
# ------------------------------------------------------------
def _score_higher_is_better(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0
    if value >= high:
        return 100
    return (value - low) / (high - low) * 100


def _score_lower_is_better(value: float, low: float, high: float) -> float:
    if value <= low:
        return 100
    if value >= high:
        return 0
    return (high - value) / (high - low) * 100


# ------------------------------------------------------------
# Score a Single KPI or Sentiment Metric
# ------------------------------------------------------------
def score_kpi(name: str, value: Optional[float]) -> Optional[float]:
    """
    Returns FHI score (0–100) for an individual KPI or sentiment metric.
    """
    if value is None:
        return None

    # Loop through dimensions in YAML (profitability/liquidity/solvency/growth/sentiment)
    for dim, metrics in THRESHOLDS.items():
        if name in metrics:
            cfg = metrics[name]
            if cfg["type"] == "higher_better":
                return _score_higher_is_better(value, cfg["low"], cfg["high"])
            else:
                return _score_lower_is_better(value, cfg["low"], cfg["high"])

    return None


# ------------------------------------------------------------
# Compute Dimension Score
# ------------------------------------------------------------
def compute_dimension_score(kpis: Dict[str, Optional[float]], dim: str) -> Optional[float]:
    metrics = FHI_DIMENSIONS[dim]
    scores = [score_kpi(m, kpis.get(m)) for m in metrics if m in kpis and kpis[m] is not None]

    return None if not scores else sum(scores) / len(scores)


# ------------------------------------------------------------
# Compute Full FHI (0–100) including sentiment
# ------------------------------------------------------------
def compute_fhi(kpis: Dict[str, Optional[float]]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """
    Computes:
    - Overall FHI score (0–100)
    - Each dimension score (profitability/liquidity/solvency/growth/sentiment)
    """
    dim_scores = {}
    weighted_total = 0
    weight_sum = 0

    for dim, weight in FHI_WEIGHTS.items():
        ds = compute_dimension_score(kpis, dim)
        dim_scores[dim] = ds

        if ds is not None:
            weighted_total += ds * weight
            weight_sum += weight

    if weight_sum == 0:
        return None, dim_scores

    fhi = weighted_total / weight_sum
    return round(fhi, 2), dim_scores