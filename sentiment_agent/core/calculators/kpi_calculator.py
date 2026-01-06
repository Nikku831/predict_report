from typing import Dict, Any, Optional


def safe(x: Optional[float]) -> Optional[float]:
    if x in (None, "", "NA"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compute_kpis(
    statements: Dict[str, Dict[str, float]],
    previous: Optional[Dict[str, float]] = None,
    *,
    compound_sentiment: Optional[float],
    tone: Optional[Dict[str, float]] = None
) -> Dict[str, Optional[float]]:
    """
    Computes financial KPIs + sentiment/tone KPIs.

    ❗ Sentiment is a REQUIRED upstream dependency.
    If compound_sentiment is missing, this function MUST fail.
    """

    # -------------------------
    # Enforce sentiment dependency
    # -------------------------
    if compound_sentiment is None:
        raise ValueError(
            "compound_sentiment is required — sentiment must be computed before KPIs"
        )

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
    kpis: Dict[str, Optional[float]] = {
        # Financial
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

        # Sentiment / Tone (NO DEFAULT MASKING)
        "compound_sentiment": compound_sentiment,
        "tone_positive_ratio": None,
        "tone_negative_ratio": None,
        "uncertainty_index": None,
        "hedging_index": None,
        "confidence_index": None,
    }

    # -------------------------
    # Financial KPIs
    # -------------------------
    if revenue and cogs is not None:
        kpis["gross_margin_pct"] = (revenue - cogs) / revenue * 100

    if revenue and op_income is not None:
        kpis["operating_margin_pct"] = op_income / revenue * 100

    if revenue and net_income is not None:
        kpis["net_margin_pct"] = net_income / revenue * 100

    if current_assets and current_liabilities:
        kpis["current_ratio"] = current_assets / current_liabilities

    if (
        current_assets
        and current_liabilities
        and inventory is not None
    ):
        kpis["quick_ratio"] = (current_assets - inventory) / current_liabilities

    if total_debt and total_equity:
        kpis["debt_to_equity"] = total_debt / total_equity

    if net_income and total_equity:
        kpis["roe_pct"] = net_income / total_equity * 100

    # -------------------------
    # Growth Metrics
    # -------------------------
    if previous:
        prev_rev = safe(previous.get("revenue"))
        prev_ni = safe(previous.get("net_income"))

        if revenue and prev_rev:
            kpis["revenue_growth_yoy"] = (revenue - prev_rev) / prev_rev * 100

        if net_income and prev_ni:
            kpis["net_income_growth_yoy"] = (net_income - prev_ni) / prev_ni * 100

    # -------------------------
    # Inject Tone Metrics
    # -------------------------
    if tone:
        kpis["tone_positive_ratio"] = tone.get("tone_positive_ratio")
        kpis["tone_negative_ratio"] = tone.get("tone_negative_ratio")
        kpis["uncertainty_index"] = tone.get("uncertainty_index")
        kpis["hedging_index"] = tone.get("hedging_index")
        kpis["confidence_index"] = tone.get("confidence_index")

    return kpis
