"""Tests for the regime-based backtest engine."""

from __future__ import annotations

import pandas as pd

from src.backtest import run_regime_backtest


def test_strategy_uses_previous_day_regime_to_avoid_lookahead() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "daily_return": [0.01, 0.02, -0.03, 0.04],
            "threshold_regime": ["Low", "High", "Low", "Low"],
        }
    )

    result = run_regime_backtest(
        data,
        regime_column="threshold_regime",
        high_regime_exposure=0.5,
        transaction_cost=0.0,
    )

    assert result["strategy_position"].tolist() == [0.0, 1.0, 0.5, 1.0]
    assert result["strategy_return"].tolist() == [0.0, 0.02, -0.015, 0.04]


def test_transaction_cost_applies_to_position_change() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "daily_return": [0.00, 0.00, 0.00],
            "threshold_regime": ["Low", "High", "Low"],
        }
    )

    result = run_regime_backtest(
        data,
        regime_column="threshold_regime",
        high_regime_exposure=0.5,
        transaction_cost=0.001,
    )

    assert result["transaction_cost"].tolist() == [0.0, 0.001, 0.0005]
    assert result["strategy_return"].tolist() == [0.0, -0.001, -0.0005]
