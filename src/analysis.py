"""Regime-level analysis utilities."""

from __future__ import annotations

import pandas as pd

from src.features import TRADING_DAYS_PER_YEAR
from src.regimes import REGIME_ORDER


def regime_summary(
    data: pd.DataFrame,
    regime_column: str,
    return_column: str = "daily_return",
) -> pd.DataFrame:
    """Summarise return and risk behaviour within each regime."""

    labelled = data.dropna(subset=[regime_column, return_column]).copy()
    if labelled.empty:
        raise ValueError("No labelled return rows available for regime analysis.")

    grouped = labelled.groupby(regime_column, observed=False)[return_column]
    summary = pd.DataFrame(
        {
            "mean_daily_return": grouped.mean(),
            "annualised_volatility": grouped.std() * (TRADING_DAYS_PER_YEAR ** 0.5),
            "worst_day": grouped.min(),
            "var_5": grouped.quantile(0.05),
            "share_of_days": grouped.size() / len(labelled),
            "days": grouped.size(),
        }
    )
    return summary.reindex(REGIME_ORDER)


def regime_persistence(data: pd.DataFrame, regime_column: str) -> pd.DataFrame:
    """Measure how often each regime continues on the next labelled day."""

    labelled = data.dropna(subset=[regime_column]).sort_values("date").copy()
    if labelled.empty:
        raise ValueError("No labelled rows available for persistence analysis.")

    labelled["next_regime"] = labelled[regime_column].shift(-1)
    transitions = labelled.dropna(subset=["next_regime"])
    persistence = (
        transitions.assign(continued=transitions[regime_column] == transitions["next_regime"])
        .groupby(regime_column, observed=False)
        .agg(
            persistence_rate=("continued", "mean"),
            transitions=("continued", "size"),
        )
    )
    return persistence.reindex(REGIME_ORDER)


def build_phase3_insights(
    market_name: str,
    method_name: str,
    summary: pd.DataFrame,
    persistence: pd.DataFrame,
) -> list[str]:
    """Create concise plain-English insights from regime statistics."""

    high = summary.loc["High"]
    low = summary.loc["Low"]
    high_persistence = persistence.loc["High", "persistence_rate"]
    low_persistence = persistence.loc["Low", "persistence_rate"]

    insights = [
        (
            f"{market_name} {method_name}: High-volatility days had annualised "
            f"volatility of {high['annualised_volatility']:.2%}, versus "
            f"{low['annualised_volatility']:.2%} in Low-volatility days."
        ),
        (
            f"{market_name} {method_name}: The worst High-regime day was "
            f"{high['worst_day']:.2%}, compared with {low['worst_day']:.2%} "
            "in the Low regime."
        ),
        (
            f"{market_name} {method_name}: The 5% VaR in the High regime was "
            f"{high['var_5']:.2%}, meaning only 5% of labelled High-regime days "
            "were worse than that threshold."
        ),
        (
            f"{market_name} {method_name}: High-regime persistence was "
            f"{high_persistence:.2%}, while Low-regime persistence was "
            f"{low_persistence:.2%}; this supports the idea that volatility clusters."
        ),
    ]
    return insights
