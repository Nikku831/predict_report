from crewai.tools import BaseTool
from src.ingestion import RealTimeIngestor
from src.analysis import QuantitativeAnalyst
from src.visuals import Visualizer

ingestor = RealTimeIngestor()
analyst = QuantitativeAnalyst()
viz = Visualizer()

def get_currency_symbol(ticker: str) -> str:
    if ticker.endswith((".NS", ".BO")):
        return "₹"
    return "$"

class FetchStockDataTool(BaseTool):
    name: str = "Fetch Stock Data"
    description: str = "Fetches price history and fundamentals."

    def _run(self, ticker: str) -> str:
        # Standard: 2 Years of data for good trend detection
        hist, fund = ingestor.get_stock_data(ticker, period="2y")
        symbol = get_currency_symbol(ticker)
        return str({
            "history_summary": hist.tail().to_string(), 
            "current_price": f"{symbol}{fund.get('current_price')}",
            "pe_ratio": fund.get("pe_ratio"),
            "currency_symbol": symbol
        })

class FetchNewsTool(BaseTool):
    name: str = "Fetch Market News"
    description: str = "Searches for the latest news about a stock."

    def _run(self, ticker: str) -> str:
        return str(ingestor.get_latest_news(ticker))

class RunProphetTool(BaseTool):
    name: str = "Run Prophet Forecast"
    description: str = "Runs Prophet prediction to forecast 30 days ahead."

    def _run(self, ticker: str) -> str:
        hist, _ = ingestor.get_stock_data(ticker, period="2y")
        df = analyst.calculate_technicals(hist)
        
        # Standard: 30 Day Prediction
        price, trend, _, _ = analyst.predict_future_price(df, days_ahead=30)
        
        symbol = get_currency_symbol(ticker)
        return f"Prediction (30 Days out): {trend} at {symbol}{price:.2f}"

class GenerateChartTool(BaseTool):
    name: str = "Generate Charts"
    description: str = "Generates a prediction chart and returns the file path."

    def _run(self, ticker: str) -> str:
        hist, _ = ingestor.get_stock_data(ticker, period="2y")
        df = analyst.calculate_technicals(hist)
        # Standard: 30 Day Chart
        _, _, forecast, _ = analyst.predict_future_price(df, days_ahead=30)
        
        if forecast is None or forecast.empty:
            return "Error: Could not generate chart."

        symbol = get_currency_symbol(ticker)
        path = viz.generate_charts(df, ticker, forecast, currency_symbol=symbol)
        return f"Chart saved at: {path}"

class StockAnalysisTools:
    fetch_stock_data = FetchStockDataTool()
    fetch_news = FetchNewsTool()
    run_prophet = RunProphetTool()
    generate_chart = GenerateChartTool()