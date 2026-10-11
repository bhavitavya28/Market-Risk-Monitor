"""Run Phase 2: label volatility regimes and compare methods."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_TICKERS, MarketDataConfig, load_all_market_data
from src.features import make_feature_table
from src.regimes import add_regime_labels, regime_agreement


DISPLAY_NAMES = {"nifty50": "Nifty 50", "sp500": "S&P 500"}


def _format_percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def main() -> None:
    config = MarketDataConfig(data_dir=PROJECT_ROOT / "data", start="2008-01-01")
    raw_data = load_all_market_data(DEFAULT_TICKERS, config=config)

    print("Phase 2 completed")
    for symbol_name, raw_table in raw_data.items():
        feature_table = make_feature_table(raw_table, volatility_window=20)
        labelled, threshold_model, kmeans_model = add_regime_labels(feature_table)
        agreement = regime_agreement(labelled)

        output_file = PROJECT_ROOT / "data" / f"{symbol_name}_regimes.csv"
        labelled.to_csv(output_file, index=False)

        summary = (
            labelled.dropna(subset=["threshold_regime", "kmeans_regime"])
            .groupby(["split", "threshold_regime", "kmeans_regime"])
            .size()
            .rename("days")
            .reset_index()
        )
        summary_file = PROJECT_ROOT / "reports" / f"phase2_{symbol_name}_agreement.csv"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_file, index=False)

        print(f"\n{DISPLAY_NAMES.get(symbol_name, symbol_name)}")
        print(
            "Threshold cutoffs fitted on train only: "
            f"Low <= {threshold_model.low_cutoff:.2%}, "
            f"High > {threshold_model.high_cutoff:.2%}"
        )
        print(f"K-Means cluster mapping: {kmeans_model.cluster_to_regime}")
        print(
            "Agreement: "
            f"overall={_format_percent(agreement.get('overall'))}, "
            f"train={_format_percent(agreement.get('train'))}, "
            f"test={_format_percent(agreement.get('test'))}"
        )
        print(f"Labelled data saved to: {output_file}")
        print(f"Agreement detail saved to: {summary_file}")


if __name__ == "__main__":
    main()
