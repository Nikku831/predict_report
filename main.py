import sys
from sentiment_agent import KPI_FHI_Crew
# Import the ingestion module
from sentiment_agent.ingestion import fetch_company_data

def run():
    # 1. Configuration
    # Use "AAPL" for US stocks or "RELIANCE.NS" for Indian stocks.
    # The ingestion script now handles the INR conversion automatically.
    TARGET_COMPANY = "IDEA.NS" 
    
    # 2. Automated Ingestion
    print(f"\n--- Starting Automated Ingestion for {TARGET_COMPANY} ---")
    try:
        # This fetches statements, history, and news automatically
        payload = fetch_company_data(TARGET_COMPANY)
        
        # Validation print to ensure data is there
        print(f"Data Fetched Successfully!")
        
        # Optional: Print Revenue to verify it looks correct (e.g. Trillions for INR)
        rev = payload['statements']['income_statement']['revenue']
        print(f"Revenue (INR): {rev:,.2f}")
        print(f"News Chars: {len(payload['earnings_call']['transcript'])}")
        
    except Exception as e:
        print(f"CRITICAL ERROR during ingestion: {e}")
        sys.exit(1)

    # 3. Run the Agent Crew
    print("\n--- Initializing Agent Crew ---")
    crew = KPI_FHI_Crew(verbose=True)
    result = crew.run(payload)
    
    print("\n\n########################")
    print("## FINAL ANALYSIS RESULT ##")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    run()