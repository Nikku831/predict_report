import pandas as pd
import numpy as np
from prophet import Prophet
import logging

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class QuantitativeAnalyst:
    def calculate_technicals(self, df):
        if df.empty: return df
        df = df.copy()
        
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()

    def predict_future_price(self, df, days_ahead=30):
        """
        Forecasting Model: Meta's Prophet.
        Predicts 30 days into the future (default).
        """
        # Data sufficiency check (20 days minimum)
        if df.empty or len(df) < 20: 
            return 0.0, "INCONCLUSIVE", pd.DataFrame(), None
        
        print(f"📈 Running Prophet Forecast ({days_ahead} days ahead)...")
        
        prophet_df = df[['Close']].reset_index()
        prophet_df.columns = ['ds', 'y']
        
        if prophet_df['ds'].dt.tz is not None:
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        target_price = forecast['yhat'].iloc[-1]
        current_price = df['Close'].iloc[-1]
        trend = "BULLISH" if target_price > current_price else "BEARISH"
        
        return target_price, trend, forecast, model