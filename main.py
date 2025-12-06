import os
from crewai import Agent, Task, Crew, Process, LLM
from crew_tools import StockAnalysisTools
from src.visuals import PDFGenerator
from dotenv import load_dotenv

load_dotenv()
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found in .env")
    exit(1)

# Using native CrewAI LLM class with 'gemini/' prefix
llm = LLM(
    model="gemini/gemini-2.5-flash",
    verbose=True,
    temperature=0.3,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Agents ---
market_researcher = Agent(
    role='Senior Market Researcher',
    goal='Gather comprehensive financial data and news.',
    backstory="You are an expert at digging up financial statements and market news.",
    verbose=True,
    tools=[StockAnalysisTools.fetch_stock_data, StockAnalysisTools.fetch_news],
    llm=llm
)

quant_analyst = Agent(
    role='Quantitative Analyst',
    goal='Forecast stock trends for the next 30 days.',
    backstory="You use Prophet models to predict medium-term stock trends.",
    verbose=True,
    tools=[StockAnalysisTools.run_prophet, StockAnalysisTools.generate_chart],
    llm=llm
)

report_writer = Agent(
    role='Equity Research Analyst',
    goal='Write a professional research report with a 30-day outlook.',
    backstory="You write equity research reports for hedge funds. You MUST use Markdown headers (#) and bold text (**) for formatting.",
    verbose=True,
    llm=llm
)

# --- Tasks ---
task_gather_data = Task(
    description="Fetch stock fundamentals, price history, and top news headlines for {ticker}.",
    agent=market_researcher,
    expected_output="A summary of the stock's financial health and news."
)

task_analyze = Task(
    description="Run the Prophet model to predict the price 30 days out for {ticker}. Also generate a chart.",
    agent=quant_analyst,
    expected_output="A technical forecast stating the Trend and predicted price."
)

task_write_report = Task(
    description="Write a Comprehensive Research Report for {ticker}. Include Current Price, 30-Day Prediction, and analysis of news. Use headers and bold text.",
    agent=report_writer,
    expected_output="A structured markdown report."
)

# --- Crew ---
stock_crew = Crew(
    agents=[market_researcher, quant_analyst, report_writer],
    tasks=[task_gather_data, task_analyze, task_write_report],
    verbose=True,
    process=Process.sequential
)

if __name__ == "__main__":
    print("--- CrewAI Stock Research Team (Standard 30-Day Model) ---")
    ticker = input("Enter Ticker (e.g., TSLA, RELIANCE.NS): ").strip().upper()
    
    result_obj = stock_crew.kickoff(inputs={'ticker': ticker})
    final_report_text = str(result_obj)

    print("\n\n########################")
    print("## GENERATING PDF...  ##")
    print("########################\n")

    pdf_gen = PDFGenerator()
    pdf_file = pdf_gen.create_crew_pdf(final_report_text, ticker)
    
    print(f"✅ Report saved successfully: {pdf_file}")