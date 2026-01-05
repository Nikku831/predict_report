from typing import Dict, Any, Optional

def safe(x: Optional[float]) -> Optional[float]:
    return None if x in (None, "", "NA") else float(x)


def compute_kpis(
    statements: Dict[str, Dict[str, float]],
    previous: Optional[Dict[str, float]] = None,
    sentiment: Optional[Dict[str, float]] = None,
    tone: Optional[Dict[str, float]] = None
) -> Dict[str, Optional[float]]:
    """
    Computes financial KPIs AND sentiment/tone KPIs.
    This version ensures sentiment outputs are NEVER null.
    """

    # -------------------------
    # Extract financial data
    # -------------------------
    is_ = statements.get("income_statement", {})
    bs_ = statements.get("balance_sheet", {})

    revenue = safe(is_.get("revenue"))
    cogs = safe(is_.get("cogs"))
    op_income = safe(is_.get("operating_income"))
    net_income = safe(is_.get("net_income"))
    total_equity = safe(bs_.get("total_equity"))
    total_debt = safe(bs_.get("total_debt"))
    current_assets = safe(bs_.get("current_assets"))
    current_liabilities = safe(bs_.get("current_liabilities"))
    inventory = safe(bs_.get("inventory"))

    # -------------------------
    # Initialize KPI dict
    # -------------------------
    kpis = {
        "revenue": revenue,
        "gross_margin_pct": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "current_ratio": None,
        "quick_ratio": None,
        "debt_to_equity": None,
        "roe_pct": None,
        "revenue_growth_yoy": None,
        "net_income_growth_yoy": None,

        # sentiment + tone KPIs
        "compound_sentiment": 0.0,
        "tone_positive_ratio": 0.0,
        "tone_negative_ratio": 0.0,
        "uncertainty_index": 0.0,
        "hedging_index": 0.0,
        "confidence_index": 0.0
    }

    # -------------------------
    # Financial KPIs
    # -------------------------
    if revenue and cogs is not None:
        kpis["gross_margin_pct"] = (revenue - cogs) / revenue * 100

    if revenue and op_income:
        kpis["operating_margin_pct"] = op_income / revenue * 100

    if revenue and net_income:
        kpis["net_margin_pct"] = net_income / revenue * 100

    if current_assets and current_liabilities:
        kpis["current_ratio"] = current_assets / current_liabilities

    if current_assets and current_liabilities and inventory is not None:
        kpis["quick_ratio"] = (current_assets - inventory) / current_liabilities

    if total_debt and total_equity:
        kpis["debt_to_equity"] = total_debt / total_equity

    if net_income and total_equity:
        kpis["roe_pct"] = net_income / total_equity * 100

    # -------------------------
    # Growth Metrics
    # -------------------------
    if previous:
        prev_rev = previous.get("revenue")
        prev_ni = previous.get("net_income")

        if revenue and prev_rev:
            kpis["revenue_growth_yoy"] = (revenue - prev_rev) / prev_rev * 100

        if net_income and prev_ni:
            kpis["net_income_growth_yoy"] = (net_income - prev_ni) / prev_ni * 100

    # -------------------------
    # Inject Sentiment
    # -------------------------
    if sentiment:
        kpis["compound_sentiment"] = float(sentiment.get("compound_sentiment", 0.0))

    # -------------------------
    # Inject Tone Metrics
    # -------------------------
    if tone:
        kpis["tone_positive_ratio"] = tone.get("tone_positive_ratio", 0.0)
        kpis["tone_negative_ratio"] = tone.get("tone_negative_ratio", 0.0)
        kpis["uncertainty_index"] = tone.get("uncertainty_index", 0.0)
        kpis["hedging_index"] = tone.get("hedging_index", 0.0)
        kpis["confidence_index"] = tone.get("confidence_index", 0.0)

    return kpis