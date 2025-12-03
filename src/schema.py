from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class StockDataSchema:
    ticker: str
    current_price: float
    market_cap: int
    pe_ratio: float
    
    # Technical Indicators
    sma_50: float
    sma_200: float
    rsi: float
    
    # AI Prediction
    prediction_30d: float
    trend: str
    
    # Qualitative Data
    news_summary: str
    
    # Market Context (Added for India Support)
    currency_symbol: str = "$" 
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a dictionary for the LLM."""
        return asdict(self)