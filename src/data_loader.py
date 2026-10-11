"""Data download, caching, and cleaning utilities.

The functions in this module are deliberately small and boring: market data
pipelines should be easy to inspect because every later result depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS: dict[str, str] = {
    "nifty50": "^NSEI",
    "sp500": "^GSPC",
}


@dataclass(frozen=True)
class MarketDataConfig:
    """Configuration for loading daily market data."""

    data_dir: Path = Path("data")
    start: str = "2008-01-01"
    end: str | None = None


def _cache_path(symbol_name: str, data_dir: Path) -> Path:
    """Return the CSV path used for a named market symbol."""

    return data_dir / f"{symbol_name}.csv"


def _standardize_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a yfinance response into a clean single-index OHLCV table."""

    if raw.empty:
        raise ValueError(f"No rows returned for {ticker}.")

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns = [str(col).strip().lower().replace(" ", "_") for col in data.columns]

    if "date" not in data.columns:
        raise ValueError(f"Downloaded data for {ticker} does not contain a date column.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None)
    data = data.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

    keep_columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    available_columns = [col for col in keep_columns if col in data.columns]
    data = data[available_columns]

    price_column = "adj_close" if "adj_close" in data.columns else "close"
    data = data.dropna(subset=[price_column])

    numeric_columns = [col for col in data.columns if col != "date"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")

    return data.reset_index(drop=True)


def download_market_data(ticker: str, config: MarketDataConfig) -> pd.DataFrame:
    """Download daily data for one ticker from Yahoo Finance."""

    raw = yf.download(
        ticker,
        start=config.start,
        end=config.end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _standardize_download(raw, ticker)


def load_market_data(
    symbol_name: str,
    ticker: str,
    config: MarketDataConfig | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load one market's data from cache, downloading it when needed.

    If Yahoo Finance fails but a cache exists, the cached file is used. That
    keeps the project usable offline after the first successful run.
    """

    config = config or MarketDataConfig()
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(symbol_name, data_dir)

    if cache_file.exists() and not force_download:
        return pd.read_csv(cache_file, parse_dates=["date"])

    try:
        data = download_market_data(ticker, config)
    except Exception:
        if cache_file.exists():
            return pd.read_csv(cache_file, parse_dates=["date"])
        raise

    data.to_csv(cache_file, index=False)
    return data


def load_all_market_data(
    tickers: dict[str, str] | None = None,
    config: MarketDataConfig | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load all configured markets into a dictionary keyed by friendly name."""

    tickers = tickers or DEFAULT_TICKERS
    return {
        symbol_name: load_market_data(symbol_name, ticker, config, force_download)
        for symbol_name, ticker in tickers.items()
    }
