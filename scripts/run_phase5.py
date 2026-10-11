"""Run Phase 5: out-of-sample validation and sensitivity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_regime_backtest
from src.data_loader import DEFAULT_TICKERS, MarketDataConfig, load_all_market_data
from src.features import make_feature_table
from src.metrics import performance_summary
from src.regimes import add_regime_labels


DISPLAY_NAMES = {"nifty50": "Nifty 50", "sp500": "S&P 500"}
REGIME_METHODS = {
    "threshold_regime": "Threshold",
    "kmeans_regime": "K-Means",
}
VOLATILITY_WINDOWS = [10, 20, 60]
HIGH_REGIME_EXPOSURES = [0.25, 0.50, 0.75]


def split_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    """Compute strategy comparison metrics separately for train and test rows."""

    frames = []
    for split_name in ["train", "test"]:
        split_backtest = backtest.loc[backtest["split"] == split_name].copy()
        # Recompute equity inside each split so test drawdown starts from the
        # first test day, not from a peak reached during the training period.
        metrics = pd.DataFrame(
            {
                "Buy and hold": performance_summary(split_backtest["baseline_return"]),
                "Risk-scaled strategy": performance_summary(split_backtest["strategy_return"]),
            }
        ).T
        metrics.insert(0, "split", split_name)
        metrics.insert(1, "strategy", metrics.index)
        frames.append(metrics.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def evaluate_configuration(
    raw_table: pd.DataFrame,
    volatility_window: int,
    regime_column: str,
    high_regime_exposure: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit train-only regimes, backtest, and return labelled data plus metrics."""

    feature_table = make_feature_table(raw_table, volatility_window=volatility_window)
    labelled, _, _ = add_regime_labels(
        feature_table,
        volatility_column=f"volatility_{volatility_window}d",
    )
    backtest = run_regime_backtest(
        labelled,
        regime_column=regime_column,
        high_regime_exposure=high_regime_exposure,
        transaction_cost=0.001,
    )
    metrics = split_metrics(backtest)
    return backtest, metrics


def run_sensitivity(raw_table: pd.DataFrame, symbol_name: str) -> pd.DataFrame:
    """Run the volatility-window and exposure-cut sensitivity grid."""

    rows = []
    for volatility_window in VOLATILITY_WINDOWS:
        for regime_column, method_name in REGIME_METHODS.items():
            for exposure in HIGH_REGIME_EXPOSURES:
                _, metrics = evaluate_configuration(
                    raw_table,
                    volatility_window=volatility_window,
                    regime_column=regime_column,
                    high_regime_exposure=exposure,
                )
                for _, row in metrics.iterrows():
                    row_data = row.to_dict()
                    row_data.update(
                        {
                            "market": symbol_name,
                            "volatility_window": volatility_window,
                            "regime_method": method_name,
                            "high_regime_exposure": exposure,
                        }
                    )
                    rows.append(row_data)
    return pd.DataFrame(rows)


def _format_percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _format_number(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def print_core_results(market_name: str, method_name: str, metrics: pd.DataFrame) -> None:
    """Print compact train/test metrics for the default Phase 5 setup."""

    print(f"\n{market_name} - {method_name} regimes")
    display_columns = ["split", "strategy", "cagr", "annualised_volatility", "sharpe_ratio", "max_drawdown"]
    formatted = metrics[display_columns].copy()
    for column in ["cagr", "annualised_volatility", "max_drawdown"]:
        formatted[column] = formatted[column].map(_format_percent)
    formatted["sharpe_ratio"] = formatted["sharpe_ratio"].map(_format_number)
    print(formatted.to_string(index=False))


def build_honest_assessment(sensitivity: pd.DataFrame) -> list[str]:
    """Summarise whether the strategy helped in train and test."""

    strategy_rows = sensitivity[sensitivity["strategy"] == "Risk-scaled strategy"].copy()
    baseline_rows = sensitivity[sensitivity["strategy"] == "Buy and hold"].copy()
    merged = strategy_rows.merge(
        baseline_rows,
        on=["market", "volatility_window", "regime_method", "high_regime_exposure", "split"],
        suffixes=("_strategy", "_baseline"),
    )
    merged["drawdown_improved"] = merged["max_drawdown_strategy"] > merged["max_drawdown_baseline"]
    merged["sharpe_improved"] = merged["sharpe_ratio_strategy"] > merged["sharpe_ratio_baseline"]
    merged["cagr_improved"] = merged["cagr_strategy"] > merged["cagr_baseline"]

    assessment = []
    for split_name, split_data in merged.groupby("split"):
        assessment.append(
            f"{split_name}: drawdown improved in {split_data['drawdown_improved'].mean():.0%} "
            f"of sensitivity cases, Sharpe improved in {split_data['sharpe_improved'].mean():.0%}, "
            f"and CAGR improved in {split_data['cagr_improved'].mean():.0%}."
        )
    return assessment


def main() -> None:
    config = MarketDataConfig(data_dir=PROJECT_ROOT / "data", start="2008-01-01")
    raw_data = load_all_market_data(DEFAULT_TICKERS, config=config)
    all_sensitivity = []

    print("Phase 5 completed")
    for symbol_name, raw_table in raw_data.items():
        market_name = DISPLAY_NAMES.get(symbol_name, symbol_name)
        market_sensitivity = run_sensitivity(raw_table, symbol_name)
        all_sensitivity.append(market_sensitivity)

        for regime_column, method_name in REGIME_METHODS.items():
            _, metrics = evaluate_configuration(
                raw_table,
                volatility_window=20,
                regime_column=regime_column,
                high_regime_exposure=0.50,
            )
            metrics_file = PROJECT_ROOT / "reports" / f"phase5_{symbol_name}_{regime_column}_train_test_metrics.csv"
            metrics.to_csv(metrics_file, index=False)
            print_core_results(market_name, method_name, metrics)
            print(f"Saved train/test metrics: {metrics_file}")

        sensitivity_file = PROJECT_ROOT / "reports" / f"phase5_{symbol_name}_sensitivity.csv"
        market_sensitivity.to_csv(sensitivity_file, index=False)
        print(f"Saved sensitivity grid: {sensitivity_file}")

    sensitivity = pd.concat(all_sensitivity, ignore_index=True)
    combined_file = PROJECT_ROOT / "reports" / "phase5_sensitivity_all.csv"
    sensitivity.to_csv(combined_file, index=False)

    assessment = build_honest_assessment(sensitivity)
    assessment_file = PROJECT_ROOT / "reports" / "phase5_assessment.txt"
    assessment_file.write_text("\n".join(f"- {line}" for line in assessment), encoding="utf-8")

    print(f"\nSaved combined sensitivity: {combined_file}")
    print(f"Saved honest assessment: {assessment_file}")
    print("\nHonest assessment")
    for line in assessment:
        print(f"- {line}")


if __name__ == "__main__":
    main()
