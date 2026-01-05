from typing import Dict, Any, List

# --------------------------------------------------
# RELATIVE IMPORTS
# --------------------------------------------------
from ..core.calculators.kpi_calculator import compute_kpis
from ..core.fhi import compute_fhi
from ..core.sentiment_model import analyze_sentiment
from ..core.tone_rules import compute_tone_metrics

class KPI_FHI_SentimentAgent:
    """
    Deterministic agent.
    Logic restored to original: KPIs computed first.
    """

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # ----------------------------
        # Validate payload
        # ----------------------------
        required_keys = ["company_id", "period", "statements", "earnings_call"]
        for key in required_keys:
            if key not in payload:
                raise ValueError(f"Missing required field: {key}")

        company_id = payload["company_id"]
        period = payload["period"]

        # ----------------------------
        # 1. Compute KPIs (Original Order)
        # ----------------------------
        statements = payload["statements"]
        historical = payload.get("historical_kpis", [])

        prev_kpis = historical[-1]["kpis"] if historical else None
        kpis = compute_kpis(statements, previous=prev_kpis)

        # ----------------------------
        # 2. Compute FHI
        # ----------------------------
        fhi_value, dim_scores = compute_fhi(kpis)
        fhi = {
            "score": fhi_value,
            "dimension_scores": dim_scores,
        }

        # ----------------------------
        # 3. Sentiment Analysis
        # ----------------------------
        transcript = payload["earnings_call"].get("transcript", "")
        sentiment = analyze_sentiment(transcript)

        # ----------------------------
        # 4. Tone Analysis
        # ----------------------------
        tone = compute_tone_metrics(transcript)

        # ----------------------------
        # 5. Risk Flags
        # ----------------------------
        risk_flags: List[str] = []

        if tone.get("uncertainty_index", 0.0) > 0.01:
            risk_flags.append("High uncertainty")

        if tone.get("hedging_index", 0.0) > 0.01:
            risk_flags.append("Management hedging language detected")

        if sentiment.get("label") == "negative":
            risk_flags.append("Negative management tone")

        # ----------------------------
        # Final output
        # ----------------------------
        return {
            "company_id": company_id,
            "period": period,
            "kpis": kpis,
            "fhi": fhi,
            "sentiment": sentiment,
            "tone": tone,
            "risk_flags": risk_flags,
        }

def run_kpi_sentiment_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Instantiates the class directly to avoid CrewAI API Key check
    agent = KPI_FHI_SentimentAgent()
    return agent.run(payload)