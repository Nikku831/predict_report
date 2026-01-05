# sentiment_agent/integration/recommendation_adapter.py

def adapt_for_recommendation(sentiment_output: dict) -> dict:
    """
    Converts sentiment_agent output into recommendation-agent input schema
    """
    return {
        "company": sentiment_output.get("company_id"),
        "financial_health": sentiment_output["fhi"]["score"],
        "sentiment_score": sentiment_output["sentiment"]["compound_sentiment"],
        "risk_flags": sentiment_output.get("risk_flags", []),
        "kpis": sentiment_output.get("kpis", {}),
        "period": sentiment_output.get("period")
    }