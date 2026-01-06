import os
import json
import time
import re
from crewai import Agent, Task, Crew, Process, LLM
from crew_tools import StockAnalysisTools
from src.visuals import PDFGenerator
from dotenv import load_dotenv

# --- INTEGRATION IMPORTS ---
from sentiment_agent.ingestion import fetch_company_data
from sentiment_agent.crew import KPI_FHI_Crew

load_dotenv()
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️ Note: GOOGLE_API_KEY not found. Ensure you have your LLM API key set.")

llm = LLM(
    model="gemini/gemini-flash-latest",
    verbose=True,
    temperature=0.3,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- AGENTS ---
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
    goal='Write a professional research report including Sentiment, FHI scores, and Recommendations.',
    backstory="You write equity research reports. You integrate quantitative data (FHI, Sentiment) with qualitative news analysis.",
    verbose=True,
    llm=llm
)

# --- CLEANING FUNCTION (UPDATED) ---
def clean_llm_output(text):
    """
    Cleans the LLM output by removing:
    1. Internal thoughts/plans
    2. Safety disclaimers
    3. Markdown artifacts
    """
    # 1. Try Delimiter Extraction (Best Case)
    pattern = r"---REPORT_START---(.*?)---REPORT_END---"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Fallback: Find "Executive Summary"
        header_pattern = r"(?m)^(?:#+|\*\*|1\.)\s*Executive Summary"
        match_header = re.search(header_pattern, text)
        if match_header:
            text = text[match_header.start():].strip()

    # 2. General Cleanup (Thought Process)
    text = re.sub(r'(?i)^(I must|Drafting|Plan:|Thought:|Mandatory Data:|Structure:).*', '', text, flags=re.MULTILINE)

    # 3. REMOVE DISCLAIMERS (The Fix)
    # Removes lines starting with "Disclaimer:" or "*Disclaimer:"
    text = re.sub(r'(?i)^\*?\s*Disclaimer:.*', '', text, flags=re.MULTILINE)
    
    return text.strip()

def run_integrated_pipeline(ticker):
    print(f"\n🚀 Starting Integrated Pipeline for {ticker}...")

    # ---------------------------------------------------------
    # STEP 1: Run Sentiment Agent
    # ---------------------------------------------------------
    print("--- 1. Running Sentiment & Financial Health Engine ---")
    
    fhi_score = "N/A"
    sentiment_score = 0
    rating = "N/A"
    risk = "Unknown"
    real_payload = {}
    sentiment_results = None

    try:
        print("   -> Fetching real financial data...")
        real_payload = fetch_company_data(ticker)
        
        sentiment_crew = KPI_FHI_Crew(verbose=True)
        sentiment_results = sentiment_crew.run(real_payload)
        
        analysis = sentiment_results.get('analysis', {})
        rec = sentiment_results.get('recommendation', {})
        reasoning = rec.get('reasoning', {})

        fhi_score = analysis.get('fhi', {}).get('score', "N/A")
        sentiment_score = analysis.get('sentiment', {}).get('compound', 0)
        rating = rec.get('rating', "N/A")
        risk = rec.get('risk_level') or reasoning.get('risk_level', "Unknown")
        
        print(f"   ✅ Engine Results: FHI={fhi_score}, Rating={rating}, Risk={risk}")

    except Exception as e:
        print(f"   ⚠️ Sentiment Engine Failed: {e}")

    # ---------------------------------------------------------
    # STEP 2: Run Research Crew
    # ---------------------------------------------------------
    print("\n--- 2. Running Research Crew (Gemini) ---")
    
    task_analyze_data = Task(
        description=f"""
        Analyze the following Real Financial Data for {ticker}:
        {json.dumps(real_payload.get('statements', {}), indent=2)}
        News: {real_payload.get('earnings_call', {}).get('transcript', 'No news found')}
        """,
        agent=market_researcher,
        expected_output="A structured summary of financial data and news."
    )

    task_forecast = Task(
        description=f"Run the Prophet model to predict 30-day price for {ticker}.",
        agent=quant_analyst,
        expected_output="A technical forecast."
    )

    task_write_report = Task(
        description=f"""
        Write a Professional Equity Research Report for {ticker}.
        
        CRITICAL INSTRUCTION:
        You MUST start your final report output with the tag "---REPORT_START---" and end it with "---REPORT_END---".
        
        Inside the tags, write ONLY the final report content.
        Do NOT include "Disclaimer:", "Thought:", "Plan:", or lists of data.
        
        MANDATORY DATA TO INTEGRATE (Write these into paragraphs, do NOT list them):
        - FHI Score: {fhi_score}/100
        - Sentiment: {sentiment_score:.2f}
        - Rating: {rating}
        - Risk Level: {risk}
        
        Report Sections:
        1. Executive Summary (Include the Rating)
        2. Financial Health & Sentiment
        3. Market Analysis
        4. Technical Outlook (30-Day Forecast)
        5. Final Recommendation
        """,
        agent=report_writer,
        expected_output="A report wrapped in ---REPORT_START--- and ---REPORT_END--- tags."
    )

    stock_crew = Crew(
        agents=[market_researcher, quant_analyst, report_writer],
        tasks=[task_analyze_data, task_forecast, task_write_report],
        verbose=True,
        process=Process.sequential
    )

    result_obj = stock_crew.kickoff(inputs={'ticker': ticker})
    
    # ---------------------------------------------------------
    # STEP 3: Generate PDF
    # ---------------------------------------------------------
    print("\n--- 3. Generating Enhanced PDF ---")
    try:
        raw_text = str(result_obj)
        final_report_text = clean_llm_output(raw_text)
        
        pdf_gen = PDFGenerator()
        pdf_file = pdf_gen.create_crew_pdf(final_report_text, ticker, sentiment_data=sentiment_results)
        
        print(f"✅ Integrated Report saved: {pdf_file}")
        return pdf_file
    except Exception as e:
        print(f"❌ PDF Generation Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    target_company = "SPICEJET.NS" 
    try:
        run_integrated_pipeline(target_company)
    except Exception as e:
        print(f"Pipeline Critical Failure: {e}")