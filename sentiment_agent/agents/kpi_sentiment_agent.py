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

    ✅ Correct execution order:
    1. Sentiment Analysis
    2. Tone Analysis
    3. KPI Computation (with sentiment + tone)
    4. Financial Health Index (FHI)
    5. Risk Flagging
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

        transcript = payload["earnings_call"].get("transcript", "")

        # ----------------------------
        # 1. Sentiment Analysis
        # ----------------------------
        sentiment = analyze_sentiment(transcript)

        compound_sentiment = sentiment.get("compound")
        if compound_sentiment is None:
            raise ValueError("Sentiment model did not return compound score")

        # ----------------------------
        # 2. Tone Analysis (MUST COME BEFORE KPIs)
        # ----------------------------
        tone = compute_tone_metrics(transcript)

        # ----------------------------
        # 3. Compute KPIs (with sentiment + tone)
        # ----------------------------
        statements = payload["statements"]
        historical = payload.get("historical_kpis", [])

        prev_kpis = historical[-1]["kpis"] if historical else None

        kpis = compute_kpis(
            statements=statements,
            previous=prev_kpis,
            compound_sentiment=compound_sentiment,
            tone=tone
        )

        # ----------------------------
        # 4. Compute Financial Health Index (FHI)
        # ----------------------------
        fhi_value, dim_scores = compute_fhi(kpis)
        fhi = {
            "score": fhi_value,
            "dimension_scores": dim_scores,
        }

        # ----------------------------
        # 5. Risk Flags (qualitative)
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
    """
    Direct runner (avoids CrewAI API key checks)
    """
    agent = KPI_FHI_SentimentAgent()
    return agent.run(payload)