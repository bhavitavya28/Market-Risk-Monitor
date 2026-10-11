"""Backtesting utilities for risk-scaled strategies."""

from __future__ import annotations

import pandas as pd

from src.metrics import drawdown, equity_curve, performance_summary


def make_regime_position(
    regimes: pd.Series,
    high_regime_exposure: float = 0.5,
) -> pd.Series:
    """Map previous-day regimes to position sizes used by the strategy."""

    if not 0 <= high_regime_exposure <= 1:
        raise ValueError("high_regime_exposure must be between 0 and 1.")

    position = pd.Series(1.0, index=regimes.index, dtype="float64")
    position.loc[regimes == "High"] = high_regime_exposure
    return position


def run_regime_backtest(
    data: pd.DataFrame,
    regime_column: str,
    high_regime_exposure: float = 0.5,
    transaction_cost: float = 0.001,
    return_column: str = "daily_return",
) -> pd.DataFrame:
    """Backtest buy-and-hold and risk-scaled exposure with transaction costs.

    The strategy shifts the regime by one day so today's position uses only
    information that would have been known at the previous close.
    """

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")
    required_columns = {"date", return_column, regime_column}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    output = data.sort_values("date").copy()
    output["baseline_position"] = 1.0
    output["baseline_return"] = output[return_column].fillna(0.0)

    previous_regime = output[regime_column].shift(1)
    output["strategy_position"] = make_regime_position(
        previous_regime,
        high_regime_exposure=high_regime_exposure,
    )
    # Use zero exposure before the first usable signal to avoid silently adding
    # one day of full exposure that no regime rule actually produced.
    output.loc[previous_regime.isna(), "strategy_position"] = 0.0

    output["position_change"] = output["strategy_position"].diff().abs().fillna(output["strategy_position"].abs())
    output["transaction_cost"] = output["position_change"] * transaction_cost
    output["strategy_return_before_cost"] = output["strategy_position"] * output[return_column].fillna(0.0)
    output["strategy_return"] = output["strategy_return_before_cost"] - output["transaction_cost"]

    output["baseline_equity"] = equity_curve(output["baseline_return"])
    output["strategy_equity"] = equity_curve(output["strategy_return"])
    output["baseline_drawdown"] = drawdown(output["baseline_equity"])
    output["strategy_drawdown"] = drawdown(output["strategy_equity"])
    return output


def compare_backtest_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    """Return side-by-side metrics for buy-and-hold and the strategy."""

    rows = {
        "Buy and hold": performance_summary(
            backtest["baseline_return"],
            backtest["baseline_equity"],
        ),
        "Risk-scaled strategy": performance_summary(
            backtest["strategy_return"],
            backtest["strategy_equity"],
        ),
    }
    return pd.DataFrame(rows).T
