from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class StockDataSchema:
    ticker: str
    current_price: float
    trend: str
    prediction_30d: float
    currency_symbol: str = "$"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)