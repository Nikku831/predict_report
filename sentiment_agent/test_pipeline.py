from sentiment_agent.agents.kpi_sentiment_agent import run_kpi_sentiment_agent
from sentiment_agent.integration.recommendation_adapter import adapt_for_recommendation
from sentiment_agent.recommendation.recommendation_engine import generate_recommendation

payload = {
    "company_id": "TEST_CO",
    "period": "Q4-2024",
    "statements": {
        "income_statement": {
            "revenue": 1000,
            "cogs": 400,
            "operating_income": 300,
            "net_income": 200
        },
        "balance_sheet": {
            "total_equity": 800,
            "total_debt": 200,
            "current_assets": 500,
            "current_liabilities": 250,
            "inventory": 100
        }
    },
    "earnings_call": {
        "transcript": "We achieved strong growth, strong margins, and confident forward guidance."
    },
    "peer_data": {
    "valuation_rank": 0.9,
    "profitability_rank": 0.85,
    "growth_rank": 0.8
}

}

agent_out = run_kpi_sentiment_agent(payload)
rec_in = adapt_for_recommendation(agent_out)
rec_out = generate_recommendation(rec_in)

print("COMPOUND:", agent_out["sentiment"]["compound"])
print("KPI SENTIMENT:", agent_out["kpis"]["compound_sentiment"])
print("ADAPTER SENTIMENT:", rec_in["sentiment_score"])
print("FINAL:", rec_out)
