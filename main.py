import os
from dotenv import load_dotenv
from src.ingestion import RealTimeIngestor
from src.analysis import QuantitativeAnalyst
from src.agent import ResearchAgent
from src.schema import StockDataSchema
from src.visuals import Visualizer, PDFGenerator

load_dotenv()

def main():
    print("--- Prediction and Report generation ---")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print(" CRITICAL: GOOGLE_API_KEY not found in .env file.")
        return

    ticker = input("Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS, NVDA): ").strip().upper()
    
    # 1. Determine Currency
    # If ticker ends with .NS (NSE) or .BO (BSE), use Rupee symbol
    currency_symbol = "₹" if ticker.endswith((".NS", ".BO")) else "$"

    # 2. Ingestion Phase
    ingestor = RealTimeIngestor()
    hist_data, fundamentals = ingestor.get_stock_data(ticker, period="3y") 
    news_data = ingestor.get_latest_news(ticker)

    if hist_data.empty:
        print(f" No data found for {ticker}. Exiting.")
        return

    # 3. Analysis Phase
    analyst = QuantitativeAnalyst()
    processed_df = analyst.calculate_technicals(hist_data)
    
    target_price, trend, forecast_df, model = analyst.predict_future_price(processed_df)
    
    print(f" Prophet 30-Day Forecast: {trend} @ {currency_symbol}{target_price:.2f}")

    # 4. Visuals Phase (Pass currency for chart labels)
    viz = Visualizer()
    chart_path = viz.generate_charts(processed_df, ticker, forecast_df, currency_symbol)

    # 5. Schema Construction
    if processed_df.empty:
        print(" Not enough data for technicals.")
        return
        
    latest_row = processed_df.iloc[-1]
    
    data_schema = StockDataSchema(
        ticker=ticker,
        current_price=fundamentals.get('current_price', 0.0),
        market_cap=fundamentals.get('market_cap', 0),
        pe_ratio=fundamentals.get('pe_ratio', 0.0),
        sma_50=latest_row['SMA_50'],
        sma_200=latest_row['SMA_200'],
        rsi=latest_row['RSI'],
        prediction_30d=round(target_price, 2),
        trend=trend,
        news_summary=str(news_data)[:1000],
        currency_symbol=currency_symbol # Pass determined currency
    )
    
    # 6. Reasoning Phase
    agent = ResearchAgent()
    report_text = agent.write_report(data_schema.to_dict())
    
    # 7. Reporting Phase
    pdf_gen = PDFGenerator()
    pdf_gen.create_pdf(data_schema.to_dict(), report_text, chart_path)
    
    print("\n Process Complete. Check the 'output' folder.")

if __name__ == "__main__":
    main()