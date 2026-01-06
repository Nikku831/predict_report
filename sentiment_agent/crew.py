# sentiment_agent/crew.py

from typing import Dict, Any

# ---- Analysis agent ----
from .agents.kpi_sentiment_agent import run_kpi_sentiment_agent

# ---- Recommendation integration ----
from .integration.recommendation_adapter import adapt_for_recommendation
from .recommendation.recommendation_engine import generate_recommendation


class KPI_FHI_Crew:
    """
    Orchestrates end-to-end financial intelligence:
    1. KPI + FHI + Sentiment +recommendation analysis
    2. Investment recommendation generation
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def run(self, payload: Dict[str, Any], mode: str = "direct") -> Dict[str, Any]:
        """
        Runs the full pipeline (DIRECT MODE).

        Expected payload format:

        {
            "company_id": "INFY",
            "period": "Q2-FY2025",
            "statements": {
                "income_statement": {...},
                "balance_sheet": {...},
                "cash_flow": {...}
            },
            "historical_kpis": [...],        # optional
            "earnings_call": {
                "transcript": "..."
            }
        }
        """

        # --------------------------------------------------
        # 1. Validation
        # --------------------------------------------------
        self._log("🔍 Validating input payload...")

        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object")

        required_fields = ["company_id", "period", "statements", "earnings_call"]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(payload["statements"], dict):
            raise ValueError(
                "'statements' must be a dictionary with financial statements"
            )

        if "transcript" not in payload["earnings_call"]:
            raise ValueError(
                "'earnings_call' must contain a 'transcript' field"
            )

        if mode != "direct":
            raise ValueError(f"Unsupported mode: {mode}")

        # --------------------------------------------------
        # 2. Sentiment + KPI + FHI Analysis
        # --------------------------------------------------
        self._log("🧠 Running KPI + FHI + Sentiment agent...")
        analysis_result = run_kpi_sentiment_agent(payload)

        # --------------------------------------------------
        # 3. Recommendation Generation
        # --------------------------------------------------
        self._log("📊 Generating investment recommendation...")
        
        rec_input = adapt_for_recommendation(analysis_result)
        rec_input["peer_data"] = payload.get("peer_data", {})
        recommendation = generate_recommendation(rec_input)

        # --------------------------------------------------
        # 4. Final Output
        # --------------------------------------------------
        self._log("✅ Full pipeline execution completed")

        return {
            "analysis": analysis_result,
            "recommendation": recommendation
        }