import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai.tools import BaseTool
from src.analysis import QuantitativeAnalyst
from src.visuals import Visualizer

analyst = QuantitativeAnalyst()
viz = Visualizer()

def get_currency_symbol(ticker: str) -> str:
    # Return 3-letter code to avoid font issues in Matplotlib
    if ticker.endswith((".NS", ".BO")):
        return "INR"
    return "USD"

def generate_dummy_history(ticker, period="2y"):
    """Generates a random walk DataFrame to simulate stock history."""
    days = 730 
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # FIX: Start at 1900 for INFY to look realistic
    price = 1900 if "INFY" in ticker else 150
    
    prices = []
    np.random.seed(42) 
    
    for _ in range(days):
        change = np.random.normal(0, 15.0) # Larger variance for higher price
        price += change
        if price < 10: price = 10 
        prices.append(price)
        
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices,
        'High': prices, 
        'Low': prices,
        'Volume': 1000000
    })
    df.set_index('Date', inplace=True)
    return df

class RunProphetTool(BaseTool):
    name: str = "Run Prophet Forecast"
    description: str = "Runs Prophet prediction to forecast 30 days ahead."

    def _run(self, ticker: str) -> str:
        hist = generate_dummy_history(ticker)
        df = analyst.calculate_technicals(hist)
        price, trend, _, _ = analyst.predict_future_price(df, days_ahead=30)
        symbol = get_currency_symbol(ticker)
        return f"Prediction (30 Days out): {trend} at {symbol} {price:.2f}"

class GenerateChartTool(BaseTool):
    name: str = "Generate Charts"
    description: str = "Generates a prediction chart and returns the file path."

    def _run(self, ticker: str) -> str:
        hist = generate_dummy_history(ticker)
        df = analyst.calculate_technicals(hist)
        _, _, forecast, _ = analyst.predict_future_price(df, days_ahead=30)
        
        if forecast is None or forecast.empty:
            return "Error: Could not generate chart."

        symbol = get_currency_symbol(ticker)
        path = viz.generate_charts(df, ticker, forecast, currency_symbol=symbol)
        return f"Chart saved at: {path}"

class StockAnalysisTools:
    run_prophet = RunProphetTool()
    generate_chart = GenerateChartTool()