import os
import json
import time
import re
from crewai import Agent, Task, Crew, Process, LLM
from crew_tools import StockAnalysisTools
from src.visuals import PDFGenerator
from dotenv import load_dotenv

load_dotenv()
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# --- 1. DUMMY INGESTION DATA ---
SAMPLE_INGESTION_DATA = json.dumps({
  "success": True,
  "data": {
    "ticker": "INFY.NS",
    "company_overview": {
      "company_name": "Infosys Limited",
      "sector": "Technology",
      "industry": "IT Services",
      "description": "Infosys Limited provides consulting, technology, outsourcing, and digital services globally."
    },
    "news": [
      {
        "headline": "Infosys exceeds revenue expectations",
        "summary": "Infosys Limited exceeded expectations with robust revenue growth in Q3, driven by large deal wins.",
        "date": "2025-01-14"
      },
      {
        "headline": "Strategic AI Expansion",
        "summary": "Company announces major expansion in Generative AI services.",
        "date": "2025-01-10"
      }
    ],
    "financials": {
      "income_statement": {
        "total_revenue": "19.68B",
        "net_income": "3.26B"
      },
      "balance_sheet": {
        "total_debt": "986M",
        "cash_reserves": "2.1B"
      }
    }
  }
})

if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found in .env")
    exit(1)

# --- CONFIGURING LLM ---
llm = LLM(
    model="gemini/gemini-flash-latest",
    verbose=True,
    temperature=0.3,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Agents ---
market_researcher = Agent(
    role='Senior Market Researcher',
    goal='Analyze provided financial data and news to extract key insights.',
    backstory="You are an expert at interpreting complex financial JSON data.",
    verbose=True,
    llm=llm
)

quant_analyst = Agent(
    role='Quantitative Analyst',
    goal='Forecast stock trends for the next 30 days and generate a visual chart.',
    backstory="You use Prophet models to predict medium-term stock trends.",
    verbose=True,
    tools=[StockAnalysisTools.run_prophet, StockAnalysisTools.generate_chart],
    llm=llm
)

report_writer = Agent(
    role='Equity Research Analyst',
    goal='Write a professional research report including Sentiment and Recommendations.',
    backstory="You write equity research reports. You DO NOT include 'Thought:', 'Plan:', or internal reasoning in your final output. Only the report content.",
    verbose=True,
    llm=llm
)

# --- Tasks ---
task_analyze_data = Task(
    description=f"""
    Analyze the following Ingestion Data for {{ticker}}:
    {SAMPLE_INGESTION_DATA}
    
    Extract:
    1. Key Financial Metrics.
    2. Major News Highlights.
    3. Business Overview.
    """,
    agent=market_researcher,
    expected_output="A structured summary of the provided financial data and news."
)

task_forecast = Task(
    description="Run the Prophet model to predict the price 30 days out for {ticker}. Generate the prediction chart.",
    agent=quant_analyst,
    expected_output="A technical forecast stating the Trend, predicted price, and the path to the generated chart."
)

task_write_report = Task(
    description="""
    Write a Comprehensive Research Report for {ticker} based on the analysis and forecast.
    
    The report MUST include these specific sections with Headers:
    1. **Executive Summary**
    2. **Market Analysis** (Fundamentals & News)
    3. **Technical Outlook** (30-Day Forecast)
    4. **Sentiment Analysis** (Clearly state: Bullish / Bearish / Neutral)
    5. **Recommendation** (Clearly state: BUY / SELL / HOLD with reasoning)
    
    IMPORTANT: Do NOT output your internal "Thought:" or "Plan:" steps. Just the final report.
    """,
    agent=report_writer,
    expected_output="A structured markdown report with Sentiment and Recommendation sections."
)

# --- Crew ---
stock_crew = Crew(
    agents=[market_researcher, quant_analyst, report_writer],
    tasks=[task_analyze_data, task_forecast, task_write_report],
    verbose=True,
    process=Process.sequential,
    max_rpm=5
)

# --- CLEANING FUNCTION ---
def clean_llm_output(text):
    """Removes 'Thought:', 'Plan:', and other internal agent reasoning."""
    # Remove "Thought: ... \n" blocks
    cleaned = re.sub(r'Thought:.*?(?=\n)', '', text, flags=re.IGNORECASE)
    # Remove "Plan: ... \n" blocks
    cleaned = re.sub(r'Plan:.*?(?=\n)', '', cleaned, flags=re.IGNORECASE)
    # Remove empty lines left behind
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()

def run_with_retry(ticker):
    max_retries = 3
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"\n🚀 Starting Crew Execution (Attempt {attempt+1}/{max_retries})...")
            result_obj = stock_crew.kickoff(inputs={'ticker': ticker})
            return result_obj
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"\n⚠️ Rate Limit Hit! Sleeping for 60s...")
                time.sleep(60)
                attempt += 1
            else:
                raise e
    raise Exception("❌ Max retries exceeded.")

if __name__ == "__main__":
    print("--- CrewAI Stock Research Team (Ingestion-Based) ---")
    
    input_ticker = "INFY.NS" 
    print(f"Processing Data for: {input_ticker}")
    
    try:
        result_obj = run_with_retry(input_ticker)
        
        # CLEAN THE OUTPUT BEFORE PDF GENERATION
        raw_text = str(result_obj)
        final_report_text = clean_llm_output(raw_text)

        print("\n\n########################")
        print("## GENERATING PDF...  ##")
        print("########################\n")

        pdf_gen = PDFGenerator()
        pdf_file = pdf_gen.create_crew_pdf(final_report_text, input_ticker)
        
        print(f"✅ Report saved successfully: {pdf_file}")

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")