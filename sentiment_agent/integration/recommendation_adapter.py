# sentiment_agent/integration/recommendation_adapter.py

from typing import Dict, Any


def adapt_for_recommendation(sentiment_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts sentiment_agent output into recommendation-agent input schema.

    ✅ Correctly propagates compound sentiment
    ✅ Never re-zeros sentiment
    ✅ Single source of truth enforced
    """

    # ----------------------------
    # Extract compound sentiment SAFELY
    # ----------------------------
    sentiment_block = sentiment_output.get("sentiment", {})
    compound = sentiment_block.get("compound")

    if compound is None:
        raise ValueError(
            "compound sentiment missing in sentiment_output — "
            "sentiment must be computed before recommendation"
        )

    return {
        "company": sentiment_output.get("company_id"),
        "period": sentiment_output.get("period"),

        # Financial Health Index
        "financial_health": sentiment_output["fhi"]["score"],

        # ✅ CORRECT sentiment propagation
        "sentiment_score": compound,

        # Risk & diagnostics
        "risk_flags": sentiment_output.get("risk_flags", []),

        # Full KPI context (already includes compound_sentiment)
        "kpis": sentiment_output.get("kpis", {}),
    }
