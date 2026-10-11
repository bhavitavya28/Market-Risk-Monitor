"""Tests for performance metric helpers."""

from __future__ import annotations

import math

import pandas as pd

from src.metrics import cagr, drawdown, equity_curve, max_drawdown, win_rate


def test_equity_curve_compounds_returns() -> None:
    returns = pd.Series([0.10, -0.10, 0.05])

    equity = equity_curve(returns)

    assert equity.iloc[-1] == 1.0395


def test_drawdown_and_max_drawdown() -> None:
    equity = pd.Series([1.00, 1.20, 0.90, 1.30])

    result = drawdown(equity)

    assert result.tolist() == [0.0, 0.0, -0.25, 0.0]
    assert max_drawdown(equity) == -0.25


def test_cagr_uses_equity_ratio_and_period_count() -> None:
    equity = pd.Series([1.0, 1.21])

    result = cagr(equity, periods_per_year=1)

    assert math.isclose(result, 0.21)


def test_win_rate_counts_positive_days_only() -> None:
    returns = pd.Series([0.01, 0.00, -0.02, 0.03])

    assert win_rate(returns) == 0.5
