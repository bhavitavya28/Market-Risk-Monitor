"""Run Phase 3: analyse returns, losses, VaR, and regime persistence."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import build_phase3_insights, regime_persistence, regime_summary
from src.data_loader import DEFAULT_TICKERS, MarketDataConfig, load_all_market_data
from src.features import make_feature_table
from src.regimes import add_regime_labels


DISPLAY_NAMES = {"nifty50": "Nifty 50", "sp500": "S&P 500"}
REGIME_METHODS = {
    "threshold_regime": "Threshold",
    "kmeans_regime": "K-Means",
}


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    formatted = summary.copy()
    percent_columns = [
        "mean_daily_return",
        "annualised_volatility",
        "worst_day",
        "var_5",
        "share_of_days",
    ]
    for column in percent_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    formatted["days"] = formatted["days"].map(lambda value: f"{int(value):,}")
    return formatted


def _format_persistence(persistence: pd.DataFrame) -> pd.DataFrame:
    formatted = persistence.copy()
    formatted["persistence_rate"] = formatted["persistence_rate"].map(lambda value: f"{value:.2%}")
    formatted["transitions"] = formatted["transitions"].map(lambda value: f"{int(value):,}")
    return formatted


def main() -> None:
    config = MarketDataConfig(data_dir=PROJECT_ROOT / "data", start="2008-01-01")
    raw_data = load_all_market_data(DEFAULT_TICKERS, config=config)
    all_insights: list[str] = []

    print("Phase 3 completed")
    for symbol_name, raw_table in raw_data.items():
        market_name = DISPLAY_NAMES.get(symbol_name, symbol_name)
        labelled, _, _ = add_regime_labels(make_feature_table(raw_table, volatility_window=20))

        print(f"\n{market_name}")
        for regime_column, method_name in REGIME_METHODS.items():
            summary = regime_summary(labelled, regime_column)
            persistence = regime_persistence(labelled, regime_column)
            insights = build_phase3_insights(market_name, method_name, summary, persistence)
            all_insights.extend(insights)

            summary_file = PROJECT_ROOT / "reports" / f"phase3_{symbol_name}_{regime_column}_summary.csv"
            persistence_file = PROJECT_ROOT / "reports" / f"phase3_{symbol_name}_{regime_column}_persistence.csv"
            summary.to_csv(summary_file)
            persistence.to_csv(persistence_file)

            print(f"\n{method_name} regime summary")
            print(_format_summary(summary).to_string())
            print(f"\n{method_name} persistence")
            print(_format_persistence(persistence).to_string())
            print(f"Saved: {summary_file}")
            print(f"Saved: {persistence_file}")

    insight_file = PROJECT_ROOT / "reports" / "phase3_insights.txt"
    insight_file.write_text("\n".join(f"- {insight}" for insight in all_insights), encoding="utf-8")
    print(f"\nInsights saved to: {insight_file}")


if __name__ == "__main__":
    main()
