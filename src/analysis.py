import pandas as pd
import numpy as np
from prophet import Prophet
import logging

# Suppress Prophet logging noise
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class QuantitativeAnalyst:
    def calculate_technicals(self, df):
        """Adds RSI and SMA indicators to the dataframe."""
        if df.empty:
            return df
            
        df = df.copy()
        # Simple Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI Calculation (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()

    def predict_future_price(self, df, days_ahead=30):
        """
        Forecasting Model: Meta's Prophet.
        Predicts the price 30 days into the future.
        """
        if df.empty or len(df) < 50:
            return 0.0, "INCONCLUSIVE", pd.DataFrame(), None
        
        print(" Running Prophet Forecast...")
        
        # Prepare data for Prophet
        prophet_df = df[['Close']].reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Remove timezone information (Prophet requirement)
        if prophet_df['ds'].dt.tz is not None:
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

        # Initialize model
        model = Prophet(
            yearly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(prophet_df)
        
        # Create future dates
        future = model.make_future_dataframe(periods=days_ahead)
        
        # Predict
        forecast = model.predict(future)
        
        # Extract target
        target_price = forecast['yhat'].iloc[-1]
        current_price = df['Close'].iloc[-1]
        trend = "BULLISH" if target_price > current_price else "BEARISH"
        
        # Return model object as well to allow for plotting later
        return target_price, trend, forecast, model

    def get_interactive_plot(self, model, forecast):
        """
        Generates an interactive Plotly chart if the library is installed.
        Returns a Plotly Figure object.
        """
        try:
            from prophet.plot import plot_plotly
            return plot_plotly(model, forecast)
        except ImportError:
            print(" Plotly not installed. Skipping interactive plot.")
            return None