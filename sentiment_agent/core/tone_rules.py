import re
from typing import Dict


# ============================================================
# LEXICONS FOR TONE ANALYSIS
# ============================================================

UNCERTAINTY_WORDS = [
    "may", "might", "could", "potentially", "uncertain", "uncertainty",
    "volatility", "challenging", "headwinds", "risk", "risks", "unpredictable"
]

HEDGING_PHRASES = [
    "we believe", "we think", "we expect", "we anticipate",
    "it is possible that", "likely", "should", "we cannot guarantee",
    "there can be no assurance"
]

CONFIDENCE_PHRASES = [
    "we are confident", "strong momentum", "robust demand", "on track",
    "solid performance", "well positioned", "very strong", "outperforming"
]

POSITIVE_WORDS = [
    "strong", "positive", "growth", "improving", "expanding", "profitable",
    "exceeded", "record", "optimistic", "progress", "resilient"
]

NEGATIVE_WORDS = [
    "decline", "negative", "downturn", "loss", "weak", "pressure",
    "slowdown", "missed", "challenging", "recession", "unfavorable"
]


# ============================================================
# TEXT CLEANING & WORD MATCHING
# ============================================================

def count_word_occ(text: str, words) -> int:
    """
    Matches full words only, case-insensitive.
    """
    t = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(w)}\b", t)) for w in words)


def count_phrase_occ(text: str, phrases) -> int:
    """
    Matches multi-word phrases.
    """
    t = text.lower()
    return sum(t.count(p) for p in phrases)


# ============================================================
# MAIN TONE METRIC FUNCTION
# ============================================================

def compute_tone_metrics(text: str) -> Dict:
    """
    Returns a tone analysis dictionary that aligns perfectly with the
    sentiment scoring system and FHI model.

    Output fields:
      - tone_positive_ratio
      - tone_negative_ratio
      - uncertainty_index
      - hedging_index
      - confidence_index (optional)
    """

    words = text.split()
    total_words = max(len(words), 1)

    pos = count_word_occ(text, POSITIVE_WORDS)
    neg = count_word_occ(text, NEGATIVE_WORDS)

    tone_positive_ratio = pos / total_words
    tone_negative_ratio = neg / total_words

    uncertainty_index = count_word_occ(text, UNCERTAINTY_WORDS) / total_words
    hedging_index = count_phrase_occ(text, HEDGING_PHRASES) / total_words
    confidence_index = count_phrase_occ(text, CONFIDENCE_PHRASES) / total_words

    return {
        "tone_positive_ratio": tone_positive_ratio,
        "tone_negative_ratio": tone_negative_ratio,
        "uncertainty_index": uncertainty_index,
        "hedging_index": hedging_index,
        "confidence_index": confidence_index,
    }