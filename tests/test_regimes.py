"""Tests for volatility regime labelling."""

from __future__ import annotations

import pandas as pd

from src.regimes import add_regime_labels, fit_threshold_regime_model


def test_threshold_cutoffs_are_fit_from_training_data_only() -> None:
    train = pd.DataFrame({"volatility_20d": [0.10, 0.20, 0.30]})

    model = fit_threshold_regime_model(
        train,
        volatility_column="volatility_20d",
        low_percentile=0,
        high_percentile=100,
    )

    assert model.low_cutoff == 0.10
    assert model.high_cutoff == 0.30


def test_add_regime_labels_marks_train_and_test_chronologically() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "volatility_20d": [0.10, 0.11, 0.12, 0.13, 0.14, 0.30, 0.31, 0.32, 0.33, 0.34],
        }
    )

    labelled, _, _ = add_regime_labels(data, train_fraction=0.60)

    assert labelled["split"].tolist() == ["train"] * 6 + ["test"] * 4
    assert set(labelled["threshold_regime"].dropna()).issubset({"Low", "Medium", "High"})
    assert set(labelled["kmeans_regime"].dropna()).issubset({"Low", "Medium", "High"})
