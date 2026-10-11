"""Interactive Streamlit dashboard for the Market Risk Monitor project."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compare_backtest_metrics, run_regime_backtest
from src.data_loader import DEFAULT_TICKERS, MarketDataConfig, load_all_market_data
from src.features import make_feature_table
from src.regimes import add_regime_labels


MARKETS = {
    "Nifty 50": "nifty50",
    "S&P 500": "sp500",
}
REGIME_COLUMNS = {
    "Threshold": "threshold_regime",
    "K-Means": "kmeans_regime",
}
REGIME_COLORS = {
    "Low": "#2ca25f",
    "Medium": "#f0ad4e",
    "High": "#d95f0e",
}


st.set_page_config(page_title="Market Risk Monitor", layout="wide")


@st.cache_data(show_spinner=False)
def load_raw_market_data() -> dict[str, pd.DataFrame]:
    """Load cached market data, downloading only if the cache is missing."""

    config = MarketDataConfig(data_dir=PROJECT_ROOT / "data", start="2008-01-01")
    return load_all_market_data(DEFAULT_TICKERS, config=config)


@st.cache_data(show_spinner=False)
def build_dashboard_data(
    market_key: str,
    volatility_window: int,
    regime_method: str,
    high_regime_exposure: float,
    transaction_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build labelled data, backtest output, and metrics for selected controls."""

    raw_data = load_raw_market_data()
    feature_table = make_feature_table(
        raw_data[market_key],
        volatility_window=volatility_window,
    )
    labelled, _, _ = add_regime_labels(
        feature_table,
        volatility_column=f"volatility_{volatility_window}d",
    )
    regime_column = REGIME_COLUMNS[regime_method]
    backtest = run_regime_backtest(
        labelled,
        regime_column=regime_column,
        high_regime_exposure=high_regime_exposure,
        transaction_cost=transaction_cost,
    )
    metrics = compare_backtest_metrics(backtest)
    return backtest, metrics


def format_percent(value: float) -> str:
    """Format a decimal value as a percentage."""

    return "n/a" if pd.isna(value) else f"{value:.2%}"


def make_price_chart(data: pd.DataFrame, price_column: str, regime_column: str) -> alt.Chart:
    """Create a price chart with line segments coloured by regime."""

    chart_data = data.dropna(subset=[price_column, regime_column]).copy()
    regime_changed = chart_data[regime_column].ne(chart_data[regime_column].shift())
    chart_data["segment_id"] = regime_changed.cumsum()

    return (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{price_column}:Q", title="Adjusted close", scale=alt.Scale(zero=False)),
            color=alt.Color(
                f"{regime_column}:N",
                title="Regime",
                scale=alt.Scale(
                    domain=list(REGIME_COLORS.keys()),
                    range=list(REGIME_COLORS.values()),
                ),
            ),
            detail="segment_id:N",
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{price_column}:Q", title="Adjusted close", format=",.2f"),
                alt.Tooltip(f"{regime_column}:N", title="Regime"),
            ],
        )
        .properties(height=360)
    )


def make_equity_chart(data: pd.DataFrame) -> alt.Chart:
    """Create baseline versus strategy equity curves."""

    chart_data = data[["date", "baseline_equity", "strategy_equity"]].melt(
        id_vars="date",
        var_name="strategy",
        value_name="equity",
    )
    chart_data["strategy"] = chart_data["strategy"].map(
        {
            "baseline_equity": "Buy and hold",
            "strategy_equity": "Risk-scaled strategy",
        }
    )

    return (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("equity:Q", title="Growth of $1", scale=alt.Scale(zero=False)),
            color=alt.Color("strategy:N", title="Strategy"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("strategy:N", title="Strategy"),
                alt.Tooltip("equity:Q", title="Equity", format=",.2f"),
            ],
        )
        .properties(height=340)
    )


def make_drawdown_chart(data: pd.DataFrame) -> alt.Chart:
    """Create baseline versus strategy drawdown curves."""

    chart_data = data[["date", "baseline_drawdown", "strategy_drawdown"]].melt(
        id_vars="date",
        var_name="strategy",
        value_name="drawdown",
    )
    chart_data["strategy"] = chart_data["strategy"].map(
        {
            "baseline_drawdown": "Buy and hold",
            "strategy_drawdown": "Risk-scaled strategy",
        }
    )

    return (
        alt.Chart(chart_data)
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format="%")),
            color=alt.Color("strategy:N", title="Strategy"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("strategy:N", title="Strategy"),
                alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
            ],
        )
        .properties(height=300)
    )


def display_metrics(metrics: pd.DataFrame) -> None:
    """Render the metrics table with readable formatting."""

    display = metrics.reset_index(names="Strategy")
    st.dataframe(
        display,
        hide_index=True,
        column_config={
            "Strategy": st.column_config.TextColumn("Strategy", pinned=True),
            "cagr": st.column_config.NumberColumn("CAGR", format="percent"),
            "annualised_volatility": st.column_config.NumberColumn("Annualised volatility", format="percent"),
            "sharpe_ratio": st.column_config.NumberColumn("Sharpe ratio", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
            "worst_day": st.column_config.NumberColumn("Worst day", format="percent"),
            "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
        },
        alt="Performance metrics for buy-and-hold and risk-scaled strategy.",
    )


with st.sidebar:
    st.header("Controls")
    selected_market = st.selectbox("Index", list(MARKETS.keys()))
    regime_method = st.segmented_control(
        "Regime method",
        list(REGIME_COLUMNS.keys()),
        default="Threshold",
    )
    volatility_window = st.select_slider(
        "Volatility window",
        options=[10, 20, 60],
        value=20,
        help="Number of trading days used for rolling volatility.",
    )
    high_regime_exposure = st.slider(
        "Exposure in High regime",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Position size used after a High-volatility signal.",
    )
    transaction_cost_bps = st.number_input(
        "Transaction cost, bps",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="10 bps equals 0.10% per unit of position change.",
    )
    st.caption("Signals are shifted by one day to avoid lookahead bias.")


market_key = MARKETS[selected_market]
transaction_cost = transaction_cost_bps / 10_000
regime_column = REGIME_COLUMNS[regime_method]
volatility_column = f"volatility_{volatility_window}d"

st.title("Market Risk Monitor")
st.caption("Volatility regime detection and risk-scaled position sizing.")

try:
    with st.spinner("Loading market data and running backtest..."):
        dashboard_data, metrics_table = build_dashboard_data(
            market_key=market_key,
            volatility_window=volatility_window,
            regime_method=regime_method,
            high_regime_exposure=high_regime_exposure,
            transaction_cost=transaction_cost,
        )
except Exception as exc:
    st.error(f"Could not build the dashboard: {exc}")
    st.stop()

price_column = "adj_close" if "adj_close" in dashboard_data.columns else "close"
latest = dashboard_data.dropna(subset=[volatility_column, regime_column]).iloc[-1]
strategy_metrics = metrics_table.loc["Risk-scaled strategy"]
baseline_metrics = metrics_table.loc["Buy and hold"]

with st.container(horizontal=True):
    st.metric(
        "Current regime",
        str(latest[regime_column]),
        border=True,
        help="Latest labelled volatility regime.",
    )
    st.metric(
        "Latest annualised volatility",
        format_percent(float(latest[volatility_column])),
        border=True,
        help=f"{volatility_window}-day rolling volatility.",
    )
    st.metric(
        "Strategy max drawdown",
        format_percent(float(strategy_metrics["max_drawdown"])),
        delta=format_percent(float(strategy_metrics["max_drawdown"] - baseline_metrics["max_drawdown"])),
        delta_color="normal",
        border=True,
        help="Less negative is better.",
    )
    st.metric(
        "Strategy Sharpe",
        f"{strategy_metrics['sharpe_ratio']:.2f}",
        delta=f"{strategy_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']:.2f}",
        border=True,
        help="Higher is better.",
    )

st.caption(
    f"Latest date: {pd.Timestamp(latest['date']).date()} | "
    f"High-regime exposure: {high_regime_exposure:.0%} | "
    f"Transaction cost: {transaction_cost_bps:.0f} bps"
)

with st.container(border=True):
    st.subheader("Price coloured by volatility regime")
    st.altair_chart(
        make_price_chart(dashboard_data, price_column, regime_column),
        alt=f"{selected_market} price coloured by {regime_method} volatility regimes.",
    )

chart_col, table_col = st.columns([1.45, 1], gap="large")
with chart_col:
    with st.container(border=True):
        st.subheader("Strategy versus buy-and-hold")
        st.altair_chart(
            make_equity_chart(dashboard_data),
            alt="Equity curves comparing buy-and-hold with the risk-scaled strategy.",
        )
with table_col:
    with st.container(border=True):
        st.subheader("Performance metrics")
        display_metrics(metrics_table)

with st.container(border=True):
    st.subheader("Drawdown curves")
    st.altair_chart(
        make_drawdown_chart(dashboard_data),
        alt="Drawdown curves comparing buy-and-hold with the risk-scaled strategy.",
    )

with st.expander("Recent labelled rows"):
    recent_columns = [
        "date",
        price_column,
        "daily_return",
        volatility_column,
        regime_column,
        "strategy_position",
        "strategy_return",
    ]
    st.dataframe(
        dashboard_data[recent_columns].tail(30).sort_values("date", ascending=False),
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date"),
            price_column: st.column_config.NumberColumn("Adjusted close", format="%.2f"),
            "daily_return": st.column_config.NumberColumn("Daily return", format="percent"),
            volatility_column: st.column_config.NumberColumn("Annualised volatility", format="percent"),
            regime_column: st.column_config.TextColumn("Regime"),
            "strategy_position": st.column_config.NumberColumn("Strategy position", format="%.2f"),
            "strategy_return": st.column_config.NumberColumn("Strategy return", format="percent"),
        },
        alt="Most recent labelled market rows used by the dashboard.",
    )
