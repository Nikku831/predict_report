import json
from sentiment_agent import KPI_FHI_Crew

# ✅ FIXED PAYLOAD: Keys now match 'kpi_calculator.py' expectations
payload = {
  "company_id": "INFY",
  "period": "Q2-FY2025",
  "statements": {
    "income_statement": {
      "revenue": 36500,
      "net_income": 7200,        # Changed from 'net_profit'
      "operating_income": 9500,  # Added (Derived from Rev - Exp)
      "cogs": 21000              # Added (Required for Gross Margin)
    },
    "balance_sheet": {
      "total_assets": 120000,
      "total_liabilities": 45000,
      "total_equity": 75000,
      "total_debt": 20000,
      "current_assets": 40000,       # Added for Liquidity Ratios
      "current_liabilities": 30000,  # Added for Liquidity Ratios
      "inventory": 5000              # Added for Quick Ratio
    },
    "cash_flow": {
      "operating_cash_flow": 9800,
      "investing_cash_flow": -3200,
      "financing_cash_flow": -1500
    }
  },
  "earnings_call": {
    "transcript": "We achieved strong revenue growth this quarter with stable margins. We are confident in our future outlook despite some minor headwinds."
  }
}

# Run the agent
crew = KPI_FHI_Crew(verbose=True)
result = crew.run(payload)

print(json.dumps(result, indent=2))