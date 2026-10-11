"""Performance and risk metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import TRADING_DAYS_PER_YEAR


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Convert periodic returns into a cumulative equity curve."""

    clean_returns = returns.fillna(0.0)
    return initial_capital * (1.0 + clean_returns).cumprod()


def drawdown(equity: pd.Series) -> pd.Series:
    """Calculate percentage drawdown from the running equity peak."""

    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate compound annual growth rate from an equity curve."""

    clean_equity = equity.dropna()
    if len(clean_equity) < 2 or clean_equity.iloc[0] <= 0:
        return float("nan")

    years = (len(clean_equity) - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return float((clean_equity.iloc[-1] / clean_equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualised_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualised volatility from periodic returns."""

    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return float("nan")
    return float(clean_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualised Sharpe ratio using a constant risk-free rate."""

    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return float("nan")

    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - daily_risk_free
    volatility = excess_returns.std(ddof=1)
    if volatility == 0 or pd.isna(volatility):
        return float("nan")
    return float(excess_returns.mean() / volatility * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Return the worst peak-to-trough drawdown."""

    return float(drawdown(equity).min())


def win_rate(returns: pd.Series) -> float:
    """Return the share of positive-return periods."""

    clean_returns = returns.dropna()
    if clean_returns.empty:
        return float("nan")
    return float((clean_returns > 0).mean())


def performance_summary(returns: pd.Series, equity: pd.Series | None = None) -> dict[str, float]:
    """Calculate the core Phase 4 performance metrics."""

    clean_returns = returns.dropna()
    clean_equity = equity if equity is not None else equity_curve(clean_returns)
    return {
        "cagr": cagr(clean_equity),
        "annualised_volatility": annualised_volatility(clean_returns),
        "sharpe_ratio": sharpe_ratio(clean_returns),
        "max_drawdown": max_drawdown(clean_equity),
        "worst_day": float(clean_returns.min()) if not clean_returns.empty else float("nan"),
        "win_rate": win_rate(clean_returns),
    }
