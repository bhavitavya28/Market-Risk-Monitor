"""Run Phase 1: load data, compute returns/volatility, and save a chart."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_TICKERS, MarketDataConfig, load_all_market_data
from src.features import make_feature_table


EVENTS = {
    "2008 crisis": "2008-09-15",
    "2020 COVID shock": "2020-03-16",
    "2022 inflation shock": "2022-06-13",
}


def plot_volatility(feature_tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot annualised 20-day volatility and annotate major market stress dates."""

    fig, ax = plt.subplots(figsize=(13, 7))
    labels = {"nifty50": "Nifty 50", "sp500": "S&P 500"}

    for symbol_name, table in feature_tables.items():
        ax.plot(
            table["date"],
            table["volatility_20d"],
            label=labels.get(symbol_name, symbol_name),
            linewidth=1.4,
        )

    for event_name, event_date in EVENTS.items():
        date = pd.Timestamp(event_date)
        ax.axvline(date, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(
            date,
            ax.get_ylim()[1] * 0.94,
            event_name,
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
        )

    ax.set_title("20-day annualised volatility: Nifty 50 vs S&P 500")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised volatility")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    config = MarketDataConfig(data_dir=PROJECT_ROOT / "data", start="2008-01-01")
    raw_data = load_all_market_data(DEFAULT_TICKERS, config=config)
    feature_tables = {
        symbol_name: make_feature_table(table, volatility_window=20)
        for symbol_name, table in raw_data.items()
    }

    for symbol_name, table in feature_tables.items():
        output_file = PROJECT_ROOT / "data" / f"{symbol_name}_features.csv"
        table.to_csv(output_file, index=False)

    chart_path = PROJECT_ROOT / "reports" / "figures" / "phase1_volatility.png"
    plot_volatility(feature_tables, chart_path)

    print("Phase 1 completed")
    for symbol_name, table in feature_tables.items():
        first_date = table["date"].min().date()
        last_date = table["date"].max().date()
        valid_returns = int(table["daily_return"].notna().sum())
        valid_vol = int(table["volatility_20d"].notna().sum())
        latest_vol = table["volatility_20d"].dropna().iloc[-1]
        print(
            f"{symbol_name}: {len(table):,} rows, {first_date} to {last_date}, "
            f"{valid_returns:,} returns, {valid_vol:,} volatility points, "
            f"latest 20d annualised vol={latest_vol:.2%}"
        )
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
