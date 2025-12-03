import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchResults

class RealTimeIngestor:
    def __init__(self):
        # Tools for searching the web
        self.search = DuckDuckGoSearchResults()

    def get_stock_data(self, ticker: str, period="2y"):
        """Fetches historical price data and key fundamentals."""
        print(f" Ingesting price data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # 1. Get History (for the Prediction Model)
        hist = stock.history(period=period)
        
        # 2. Get Fundamentals (for the Report)
        info = stock.info
        fundamentals = {
            "current_price": info.get("currentPrice", 0.0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0.0),
            "sector": info.get("sector", "Unknown"),
        }
        
        return hist, fundamentals

    def get_latest_news(self, ticker: str):
        """Fetches top news search results."""
        print(f" Ingesting news for {ticker}...")
        query = f"{ticker} stock news analysis financial outlook"
        try:
            results = self.search.run(query)
            return results
        except Exception as e:
            return f"Error fetching news: {str(e)}"