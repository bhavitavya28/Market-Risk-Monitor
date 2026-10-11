"""Volatility regime labelling methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REGIME_ORDER = ["Low", "Medium", "High"]
RANDOM_STATE = 42


@dataclass(frozen=True)
class ThresholdRegimeModel:
    """Percentile-based volatility regime model fitted on training data."""

    low_cutoff: float
    high_cutoff: float
    volatility_column: str

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Label rows as Low, Medium, or High using fixed train cutoffs."""

        volatility = data[self.volatility_column]
        labels = pd.Series(pd.NA, index=data.index, dtype="object")
        labels.loc[volatility <= self.low_cutoff] = "Low"
        labels.loc[(volatility > self.low_cutoff) & (volatility <= self.high_cutoff)] = "Medium"
        labels.loc[volatility > self.high_cutoff] = "High"
        return labels


@dataclass
class KMeansRegimeModel:
    """K-Means volatility regime model with clusters ordered by volatility."""

    scaler: StandardScaler
    model: KMeans
    cluster_to_regime: dict[int, str]
    feature_columns: list[str]

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict ordered volatility regimes using the fitted K-Means model."""

        features = data[self.feature_columns]
        valid_mask = features.notna().all(axis=1)
        labels = pd.Series(pd.NA, index=data.index, dtype="object")

        if valid_mask.any():
            scaled_features = self.scaler.transform(features.loc[valid_mask])
            clusters = self.model.predict(scaled_features)
            labels.loc[valid_mask] = [self.cluster_to_regime[int(cluster)] for cluster in clusters]

        return labels


def chronological_train_mask(data: pd.DataFrame, train_fraction: float = 0.70) -> pd.Series:
    """Return a mask for the first train_fraction rows by date order."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    ordered = data.sort_values("date")
    train_size = int(np.floor(len(ordered) * train_fraction))
    train_index = ordered.index[:train_size]
    return data.index.isin(train_index)


def fit_threshold_regime_model(
    train_data: pd.DataFrame,
    volatility_column: str = "volatility_20d",
    low_percentile: float = 33.0,
    high_percentile: float = 67.0,
) -> ThresholdRegimeModel:
    """Fit Low/Medium/High cutoffs from training volatility percentiles only."""

    train_volatility = train_data[volatility_column].dropna()
    if train_volatility.empty:
        raise ValueError("Cannot fit threshold regimes without non-missing volatility.")

    low_cutoff, high_cutoff = np.percentile(train_volatility, [low_percentile, high_percentile])
    return ThresholdRegimeModel(
        low_cutoff=float(low_cutoff),
        high_cutoff=float(high_cutoff),
        volatility_column=volatility_column,
    )


def fit_kmeans_regime_model(
    train_data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
) -> KMeansRegimeModel:
    """Fit K-Means on training volatility features and order clusters by risk."""

    feature_columns = feature_columns or ["volatility_20d"]
    train_features = train_data[feature_columns].dropna()
    if len(train_features) < n_clusters:
        raise ValueError("Not enough non-missing rows to fit K-Means regimes.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_features)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    clusters = model.fit_predict(scaled_features)

    cluster_summary = (
        pd.DataFrame({"cluster": clusters, "volatility": train_features[feature_columns[0]].to_numpy()})
        .groupby("cluster")["volatility"]
        .mean()
        .sort_values()
    )
    cluster_to_regime = {
        int(cluster): REGIME_ORDER[position]
        for position, cluster in enumerate(cluster_summary.index)
    }

    return KMeansRegimeModel(
        scaler=scaler,
        model=model,
        cluster_to_regime=cluster_to_regime,
        feature_columns=feature_columns,
    )


def add_regime_labels(
    data: pd.DataFrame,
    train_fraction: float = 0.70,
    volatility_column: str = "volatility_20d",
) -> tuple[pd.DataFrame, ThresholdRegimeModel, KMeansRegimeModel]:
    """Fit train-only regime models and add threshold/K-Means labels."""

    output = data.copy()
    train_mask = chronological_train_mask(output.dropna(subset=[volatility_column]), train_fraction)
    train_index = output.dropna(subset=[volatility_column]).index[train_mask]
    full_train_mask = output.index.isin(train_index)

    threshold_model = fit_threshold_regime_model(
        output.loc[full_train_mask],
        volatility_column=volatility_column,
    )
    kmeans_model = fit_kmeans_regime_model(
        output.loc[full_train_mask],
        feature_columns=[volatility_column],
    )

    output["split"] = np.where(full_train_mask, "train", "test")
    output.loc[output[volatility_column].isna(), "split"] = pd.NA
    output["threshold_regime"] = threshold_model.predict(output)
    output["kmeans_regime"] = kmeans_model.predict(output)

    return output, threshold_model, kmeans_model


def regime_agreement(data: pd.DataFrame) -> dict[str, float]:
    """Return overall/train/test agreement between the two regime methods."""

    labelled = data.dropna(subset=["threshold_regime", "kmeans_regime"])
    results: dict[str, float] = {}

    for split_name, split_data in [("overall", labelled), *labelled.groupby("split")]:
        if len(split_data) == 0:
            results[str(split_name)] = float("nan")
        else:
            results[str(split_name)] = float(
                (split_data["threshold_regime"] == split_data["kmeans_regime"]).mean()
            )

    return results
