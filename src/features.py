"""Feature engineering utilities for returns and volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def add_return_features(
    data: pd.DataFrame,
    price_column: str | None = None,
) -> pd.DataFrame:
    """Add daily simple returns to a price table."""

    output = data.copy()
    selected_price_column = price_column or (
        "adj_close" if "adj_close" in output.columns else "close"
    )

    if selected_price_column not in output.columns:
        raise KeyError(f"Missing price column: {selected_price_column}")

    output["daily_return"] = output[selected_price_column].pct_change()
    return output


def add_rolling_volatility(
    data: pd.DataFrame,
    window: int = 20,
    return_column: str = "daily_return",
) -> pd.DataFrame:
    """Add annualised rolling volatility from daily returns."""

    if window < 2:
        raise ValueError("window must be at least 2 trading days.")
    if return_column not in data.columns:
        raise KeyError(f"Missing return column: {return_column}")

    output = data.copy()
    # Daily standard deviation is scaled by sqrt(252), the usual trading-day
    # convention for expressing volatility as an annualised number.
    output[f"volatility_{window}d"] = (
        output[return_column].rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    return output


def make_feature_table(
    data: pd.DataFrame,
    volatility_window: int = 20,
    price_column: str | None = None,
) -> pd.DataFrame:
    """Create the Phase 1 feature table: returns and rolling volatility."""

    output = add_return_features(data, price_column=price_column)
    output = add_rolling_volatility(output, window=volatility_window)
    return output
