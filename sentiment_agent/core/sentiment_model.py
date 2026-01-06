from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

# ----------------------------------------------------------------------
# Singleton-style model load (CRITICAL for performance)
# ----------------------------------------------------------------------
_DEVICE = 0 if torch.cuda.is_available() else -1
_MODEL_NAME = "ProsusAI/finbert"

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)

_sentiment_pipeline = pipeline(
    task="text-classification",
    model=_model,
    tokenizer=_tokenizer,
    return_all_scores=True,
    truncation=True,
    device=_DEVICE
)


class SentimentModel:
    """
    FinBERT-based sentiment engine for earnings call transcripts.

    Returns a NORMALIZED schema used across the entire pipeline:
    {
        "compound": float,
        "compound_sentiment": float,
        "positive": float,
        "neutral": float,
        "negative": float,
        "label": "positive" | "neutral" | "negative"
    }
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_text(text: str, max_chars: int = 2000) -> List[str]:
        """
        Splits long transcript into smaller chunks for stable inference.
        """
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    @staticmethod
    def _compound_score(pos: float, neg: float) -> float:
        """
        FinBERT-style compound sentiment score.
        Range: [-1, +1]
        """
        return float(pos - neg)

    @staticmethod
    def _label_from_scores(pos: float, neu: float, neg: float) -> str:
        if pos >= max(neu, neg):
            return "positive"
        if neg >= max(pos, neu):
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------
    # main inference
    # ------------------------------------------------------------------
    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "compound": 0.0,
                "compound_sentiment": 0.0,
                "positive": 0.0,
                "neutral": 1.0,
                "negative": 0.0,
                "label": "neutral"
            }

        chunks = self._chunk_text(text)
        all_pos, all_neu, all_neg = [], [], []

        # ---- Run FinBERT on each chunk ----
        for c in chunks:
            scores = _sentiment_pipeline(c)[0]

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
        label = self._label_from_scores(avg_pos, avg_neu, avg_neg)

        return {
            # ✅ canonical field used by agent & adapter
            "compound": compound,

            # ✅ backward-compatible field used by KPI calculators
            "compound_sentiment": compound,

            "positive": avg_pos,
            "neutral": avg_neu,
            "negative": avg_neg,
            "label": label
        }


# ----------------------------------------------------------------------
# Convenience wrapper (used by agents)
# ----------------------------------------------------------------------
_SENTIMENT_MODEL = SentimentModel()

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Public API used across the project.
    Guaranteed to return 'compound'.
    """
    return _SENTIMENT_MODEL.analyze(text)
