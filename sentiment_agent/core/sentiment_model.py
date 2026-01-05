from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np


class SentimentModel:
    """
    FinBERT-based sentiment engine.
    Returns:
        - compound_sentiment
        - positive / negative / neutral probabilities
    Designed for long earnings call transcripts.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            truncation=True,
            device=self.device
        )

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _chunk_text(text: str, max_chars: int = 2000) -> List[str]:
        """
        Splits long transcript into smaller pieces for stable inference.
        """
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])
        return chunks

    @staticmethod
    def _compound_score(pos: float, neg: float) -> float:
        """
        FinBERT-like compound sentiment score.
        """
        return float(pos - neg)

    # ----------------------------------------------------------------------
    # main inference function
    # ----------------------------------------------------------------------
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Returns:
           {
             "compound_sentiment": float,
             "positive": float,
             "neutral": float,
             "negative": float
           }
        """
        if not text.strip():
            return {
                "compound_sentiment": 0.0,
                "positive": 0.0,
                "neutral": 1.0,
                "negative": 0.0
            }

        chunks = self._chunk_text(text)
        all_pos, all_neu, all_neg = [], [], []

        # ---- Run FinBERT on each chunk ----
        for c in chunks:
            scores = self.sentiment_pipeline(c)[0]

            pos = next(s["score"] for s in scores if s["label"].lower() == "positive")
            neg = next(s["score"] for s in scores if s["label"].lower() == "negative")
            neu = next(s["score"] for s in scores if s["label"].lower() == "neutral")

            all_pos.append(pos)
            all_neg.append(neg)
            all_neu.append(neu)

        # ---- Aggregate across chunks ----
        avg_pos = float(np.mean(all_pos))
        avg_neg = float(np.mean(all_neg))
        avg_neu = float(np.mean(all_neu))

        compound = self._compound_score(avg_pos, avg_neg)

        return {
            "compound_sentiment": compound,
            "positive": avg_pos,
            "neutral": avg_neu,
            "negative": avg_neg
        }


# ----------------------------------------------------------------------
# Simple convenience wrapper
# ----------------------------------------------------------------------
def analyze_sentiment(text: str) -> Dict[str, Any]:
    model = SentimentModel()
    return model.analyze(text)