"""Run Phase 4: backtest baseline versus risk-scaled regime strategy."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compare_backtest_metrics, run_regime_backtest
from src.data_loader import DEFAULT_TICKERS, MarketDataConfig, load_all_market_data
from src.features import make_feature_table
from src.regimes import add_regime_labels


DISPLAY_NAMES = {"nifty50": "Nifty 50", "sp500": "S&P 500"}
REGIME_METHODS = {
    "threshold_regime": "Threshold",
    "kmeans_regime": "K-Means",
}


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    percent_columns = ["cagr", "annualised_volatility", "max_drawdown", "worst_day", "win_rate"]
    for column in percent_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    formatted["sharpe_ratio"] = formatted["sharpe_ratio"].map(lambda value: f"{value:.2f}")
    return formatted


def plot_backtest(backtest: pd.DataFrame, market_name: str, method_name: str, output_path: Path) -> None:
    """Save equity and drawdown curves for a backtest."""

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(backtest["date"], backtest["baseline_equity"], label="Buy and hold", linewidth=1.5)
    axes[0].plot(backtest["date"], backtest["strategy_equity"], label="Risk-scaled strategy", linewidth=1.5)
    axes[0].set_title(f"{market_name}: equity curves ({method_name} regimes)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(backtest["date"], backtest["baseline_drawdown"], label="Buy and hold", linewidth=1.3)
    axes[1].plot(backtest["date"], backtest["strategy_drawdown"], label="Risk-scaled strategy", linewidth=1.3)
    axes[1].set_title("Drawdowns")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    config = MarketDataConfig(data_dir=PROJECT_ROOT / "data", start="2008-01-01")
    raw_data = load_all_market_data(DEFAULT_TICKERS, config=config)

    print("Phase 4 completed")
    for symbol_name, raw_table in raw_data.items():
        market_name = DISPLAY_NAMES.get(symbol_name, symbol_name)
        labelled, _, _ = add_regime_labels(make_feature_table(raw_table, volatility_window=20))

        print(f"\n{market_name}")
        for regime_column, method_name in REGIME_METHODS.items():
            backtest = run_regime_backtest(
                labelled,
                regime_column=regime_column,
                high_regime_exposure=0.5,
                transaction_cost=0.001,
            )
            metrics = compare_backtest_metrics(backtest)

            backtest_file = PROJECT_ROOT / "reports" / f"phase4_{symbol_name}_{regime_column}_backtest.csv"
            metrics_file = PROJECT_ROOT / "reports" / f"phase4_{symbol_name}_{regime_column}_metrics.csv"
            chart_file = PROJECT_ROOT / "reports" / "figures" / f"phase4_{symbol_name}_{regime_column}.png"

            backtest.to_csv(backtest_file, index=False)
            metrics.to_csv(metrics_file)
            plot_backtest(backtest, market_name, method_name, chart_file)

            print(f"\n{method_name} regimes")
            print(_format_metrics(metrics).to_string())
            print(f"Saved backtest: {backtest_file}")
            print(f"Saved metrics: {metrics_file}")
            print(f"Saved chart: {chart_file}")


if __name__ == "__main__":
    main()
