"""
Global configuration for Recommendation Agent.

Loads JSON defaults from same folder, then overrides from environment variables
(if present). Uses python-dotenv when available to load .env file.
"""

from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent

# try to load .env automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR.parent.joinpath(".env"))
except Exception:
    pass

def _load_json(name):
    p = BASE_DIR / name
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}

# Load defaults
RATING_THRESHOLDS = _load_json("rating_thresholds.json") or {}
RISK_PENALTIES = _load_json("risk_penalties.json") or {}
WEIGHTAGES = _load_json("weightages.json") or {}

# Apply defaults if missing
RATING_THRESHOLDS.setdefault("buy_min", int(os.getenv("BUY_MIN", 75)))
RATING_THRESHOLDS.setdefault("hold_min", int(os.getenv("HOLD_MIN", 50)))
RATING_THRESHOLDS.setdefault("sell_max", int(os.getenv("SELL_MAX", 49)))

RISK_PENALTIES.setdefault("low_risk", int(os.getenv("RISK_LOW", 0)))
RISK_PENALTIES.setdefault("medium_risk", int(os.getenv("RISK_MEDIUM", -5)))
RISK_PENALTIES.setdefault("high_risk", int(os.getenv("RISK_HIGH", -15)))

WEIGHTAGES.setdefault("kpi_weight", int(os.getenv("KPI_WEIGHT", 50)))
WEIGHTAGES.setdefault("sentiment_weight", int(os.getenv("SENTIMENT_WEIGHT", 20)))
WEIGHTAGES.setdefault("peer_weight", int(os.getenv("PEER_WEIGHT", 30)))

# For convenience (export)
BUY_MIN = RATING_THRESHOLDS["buy_min"]
HOLD_MIN = RATING_THRESHOLDS["hold_min"]
SELL_MAX = RATING_THRESHOLDS["sell_max"]