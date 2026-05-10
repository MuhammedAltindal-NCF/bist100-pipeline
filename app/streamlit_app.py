from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.colors as pc
import streamlit as st


# -------------------------------------------------------
# Page configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="BIST100 Outperformance Predictor",
    page_icon="📈",
    layout="wide",
)


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bist100_gold_features.csv"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "bist100_model.pkl"
FEATURES_FILE = PROJECT_ROOT / "artifacts" / "feature_columns.json"


# -------------------------------------------------------
# Long color sequence for many tickers
# -------------------------------------------------------

def build_long_color_sequence():
    palettes = [
        pc.qualitative.Alphabet,
        pc.qualitative.Dark24,
        pc.qualitative.Light24,
        pc.qualitative.Plotly,
        pc.qualitative.G10,
        pc.qualitative.T10,
        pc.qualitative.Set3,
        pc.qualitative.Safe,
        pc.qualitative.Vivid,
        pc.qualitative.Bold,
        pc.qualitative.Prism,
        pc.qualitative.D3,
    ]

    colors = []
    for palette in palettes:
        for color in palette:
            if color not in colors:
                colors.append(color)

    return colors


LONG_COLOR_SEQUENCE = build_long_color_sequence()


# -------------------------------------------------------
# Load data and model
# -------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    with open(FEATURES_FILE, "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols


df = load_data()
model, feature_cols = load_model()


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def build_latest_predictions(data, model, feature_cols):
    latest_rows = (
        data.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )

    X_latest = latest_rows[feature_cols]
    latest_rows["predicted_probability"] = model.predict_proba(X_latest)[:, 1]
    latest_rows["predicted_class"] = model.predict(X_latest)

    latest_rows["prediction_label"] = latest_rows["predicted_class"].map(
        {1: "Outperform", 0: "Not Outperform"}
    )

    latest_rows = latest_rows.sort_values(
        "predicted_probability",
        ascending=False,
    ).reset_index(drop=True)

    return latest_rows


def build_normalized_comparison(data, selected_tickers, frequency="Monthly"):
    comparison_df = data[data["ticker"].isin(selected_tickers)].copy()
    comparison_df = comparison_df.sort_values(["ticker", "date"])

    if frequency == "Monthly":
        comparison_df["plot_date"] = (
            comparison_df["date"].dt.to_period("M").dt.to_timestamp()
        )

        comparison_df = (
            comparison_df.sort_values(["ticker", "date"])
            .groupby(["ticker", "plot_date"], as_index=False)
            .tail(1)
        )
    else:
        comparison_df["plot_date"] = comparison_df["date"]

    comparison_df = comparison_df.sort_values(["ticker", "plot_date"]).copy()

    comparison_df["normalized_close"] = (
        comparison_df.groupby("ticker")["close"]
        .transform(lambda x: (x / x.iloc[0]) * 100)
    )

    first_dates = (
        data.groupby("ticker", as_index=False)["date"]
        .min()
        .rename(columns={"date": "first_available_date"})
    )

    comparison_df = comparison_df.merge(first_dates, on="ticker", how="left")

    comparison_df["plot_date_str"] = pd.to_datetime(
        comparison_df["plot_date"]
    ).dt.strftime("%Y-%m-%d")

    comparison_df["first_available_date_str"] = pd.to_datetime(
        comparison_df["first_available_date"]
    ).dt.strftime("%Y-%m-%d")

    return comparison_df


def build_performance_summary(comparison_df):
    performance_summary = (
        comparison_df.sort_values(["ticker", "plot_date"])
        .groupby("ticker", as_index=False)
        .agg(
            first_available_date=("first_available_date", "first"),
            latest_date=("plot_date", "last"),
            start_close=("close", "first"),
            latest_close=("close", "last"),
            latest_normalized_index=("normalized_close", "last"),
        )
    )

    performance_summary["total_return_pct"] = (
        performance_summary["latest_normalized_index"] - 100
    )

    performance_summary = performance_summary.sort_values(
        "latest_normalized_index",
        ascending=False,
    ).reset_index(drop=True)

    performance_summary["first_available_date"] = pd.to_datetime(
        performance_summary["first_available_date"]
    ).dt.date

    performance_summary["latest_date"] = pd.to_datetime(
        performance_summary["latest_date"]
    ).dt.date

    performance_summary["start_close"] = performance_summary["start_close"].round(2)
    performance_summary["latest_close"] = performance_summary["latest_close"].round(2)

    performance_summary["latest_normalized_index"] = performance_summary[
        "latest_normalized_index"
    ].round(2)

    performance_summary["total_return_pct"] = performance_summary[
        "total_return_pct"
    ].round(2)

    return performance_summary


# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("BIST100 Stock Outperformance Predictor")

st.markdown(
    """
    This application predicts whether a selected BIST100 stock is likely to outperform
    the BIST100 index over the next 20 trading days.

    The model uses historical price, volume, valuation, market, and USDTRY-based features.
    """
)


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Controls")

tickers = sorted(df["ticker"].dropna().unique())
selected_ticker = st.sidebar.selectbox("Select a ticker", tickers)

latest_date = df["date"].max()
earliest_date = df["date"].min()

st.sidebar.write(f"Latest available date: **{latest_date.date()}**")
st.sidebar.write(f"Earliest available date: **{earliest_date.date()}**")
st.sidebar.write(f"Number of tickers: **{len(tickers)}**")


# -------------------------------------------------------
# Latest predictions
# -------------------------------------------------------

latest_rows = build_latest_predictions(df, model, feature_cols)
selected_latest = latest_rows[latest_rows["ticker"] == selected_ticker].iloc[0]


# -------------------------------------------------------
# Tabs
# -------------------------------------------------------

tab_single, tab_market, tab_model = st.tabs(
    [
        "Single Stock Prediction",
        "BIST100-Wide Comparison",
        "Model Notes",
    ]
)


# -------------------------------------------------------
# Tab 1: Single stock prediction
# -------------------------------------------------------

with tab_single:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Selected Ticker", selected_ticker)

    with col2:
        st.metric(
            "Predicted Probability",
            f"{selected_latest['predicted_probability']:.1%}",
        )

    with col3:
        st.metric("Prediction", selected_latest["prediction_label"])

    with col4:
        st.metric("Latest Close", f"{selected_latest['close']:.2f}")

    ticker_df = df[df["ticker"] == selected_ticker].sort_values("date").copy()

    st.subheader(f"{selected_ticker} Historical Price")

    fig_price = px.line(
        ticker_df,
        x="date",
        y="close",
        title=f"{selected_ticker} Close Price Over Time",
        template="plotly_dark",
        custom_data=["ticker", "date", "close"],
    )

    fig_price.update_traces(
        line=dict(width=2.5),
        hovertemplate=(
            "<b>Ticker:</b> %{customdata[0]}<br>"
            "<b>Date:</b> %{customdata[1]|%Y-%m-%d}<br>"
            "<b>Close Price:</b> %{customdata[2]:.2f}<br>"
            "<extra></extra>"
        ),
    )

    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price",
        hovermode="closest",
        showlegend=False,
        height=520,
    )

    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Latest Feature Values for Selected Stock")
    feature_display = selected_latest[feature_cols].to_frame(name="value")
    st.dataframe(feature_display, use_container_width=True)


# -------------------------------------------------------
# Tab 2: BIST100-wide comparison
# -------------------------------------------------------

with tab_market:
    st.subheader("BIST100-Wide Comparison")

    st.markdown(
        """
        This section compares stocks visually across time using a **multi-line comparison chart**.
        Each stock starts from **100** on its own first available date, so stocks that entered the
        dataset later appear only from the date their data actually begins.
        """
    )

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        comparison_mode = st.radio(
            "Comparison universe",
            options=[
                "Top predicted stocks",
                "Selected stocks",
                "All stocks",
            ],
            horizontal=True,
        )

    with col_b:
        chart_frequency = st.radio(
            "Chart frequency",
            options=["Monthly", "Daily"],
            horizontal=True,
            help="Monthly is recommended when comparing many stocks.",
        )

    use_log_scale = st.checkbox(
        "Use log scale for normalized index",
        value=False,
        help=(
            "Useful when a few stocks have extremely large long-term returns "
            "and make the other lines hard to read."
        ),
    )

    if comparison_mode == "Top predicted stocks":
        top_compare_n = st.slider(
            "Number of top predicted stocks to compare",
            min_value=5,
            max_value=len(tickers),
            value=min(20, len(tickers)),
        )
        comparison_tickers = latest_rows.head(top_compare_n)["ticker"].tolist()

    elif comparison_mode == "Selected stocks":
        default_selection = latest_rows.head(8)["ticker"].tolist()
        comparison_tickers = st.multiselect(
            "Select tickers to compare",
            options=tickers,
            default=default_selection,
        )

    else:
        comparison_tickers = tickers
        st.caption(
            f"All {len(tickers)} stocks are included. "
            "For readability, Monthly frequency and log scale are recommended."
        )

    if len(comparison_tickers) == 0:
        st.warning("Please select at least one ticker.")
    else:
        comparison_df = build_normalized_comparison(
            df,
            selected_tickers=comparison_tickers,
            frequency=chart_frequency,
        )

        performance_summary = build_performance_summary(comparison_df)
        top_performer = performance_summary.iloc[0]

        st.success(
            f"Historical best performer in this selected group: "
            f"{top_performer['ticker']} | "
            f"Normalized index: {top_performer['latest_normalized_index']:,.2f} | "
            f"Total return: {top_performer['total_return_pct']:,.2f}%"
        )

        st.caption(
            "Note: very high normalized returns can occur when a stock starts from a very low "
            "historical price, or when long-term price history is affected by corporate actions "
            "such as splits, bonus issues, or adjustment differences."
        )

        st.subheader("Normalized Multi-Stock Performance Chart")

        fig_compare = px.line(
            comparison_df,
            x="plot_date",
            y="normalized_close",
            color="ticker",
            title="Normalized Close Price Comparison",
            template="plotly_dark",
            color_discrete_sequence=LONG_COLOR_SEQUENCE,
            custom_data=[
                "ticker",
                "plot_date_str",
                "normalized_close",
                "close",
                "first_available_date_str",
            ],
        )

        line_width = 2.4 if len(comparison_tickers) <= 15 else 1.8
        opacity_level = 0.95 if len(comparison_tickers) <= 20 else 0.78

        fig_compare.update_traces(
            mode="lines",
            line=dict(width=line_width),
            opacity=opacity_level,
            hovertemplate=(
                "<b>Ticker:</b> %{customdata[0]}<br>"
                "<b>Date:</b> %{customdata[1]}<br>"
                "<b>Normalized Index:</b> %{customdata[2]:,.2f}<br>"
                "<b>Close Price:</b> %{customdata[3]:.2f}<br>"
                "<b>First Available Date:</b> %{customdata[4]}<br>"
                "<extra></extra>"
            ),
        )

        fig_compare.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalized Close Index (First Available Date = 100)",
            hovermode="closest",
            legend_title_text="Ticker",
            height=650,
        )

        if use_log_scale:
            fig_compare.update_yaxes(type="log")

        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown(
            """
            **How to read this chart**

            - Each colored line represents one stock.
            - Every stock starts at **100** on its own first available observation date.
            - If a stock entered the dataset later, it appears later on the chart.
            - A value of **200** means the stock doubled relative to its own starting point.
            - Hovering over a line shows only that stock's details.
            - The green box shows the strongest **historical price performer** in the selected comparison group, not the model's top prediction.
            """
        )

        st.subheader("Best Performing Stocks in the Selected Comparison")

        st.markdown(
            """
            This ranking is based on the **latest normalized index**. The stock with the highest
            latest normalized index has delivered the strongest total price performance from its
            own first available date to the latest available date.
            """
        )

        fig_perf = px.bar(
            performance_summary.head(20).sort_values(
                "latest_normalized_index",
                ascending=True,
            ),
            x="latest_normalized_index",
            y="ticker",
            orientation="h",
            title="Top 20 Historical Performers by Latest Normalized Index",
            template="plotly_dark",
            labels={
                "latest_normalized_index": "Latest Normalized Index",
                "ticker": "Ticker",
            },
            hover_data={
                "ticker": True,
                "first_available_date": True,
                "latest_date": True,
                "start_close": ":.2f",
                "latest_close": ":.2f",
                "latest_normalized_index": ":,.2f",
                "total_return_pct": ":,.2f",
            },
        )

        fig_perf.update_layout(
            xaxis_title="Latest Normalized Index",
            yaxis_title="Ticker",
            height=600,
        )

        if use_log_scale:
            fig_perf.update_xaxes(type="log")

        st.plotly_chart(fig_perf, use_container_width=True)

        st.dataframe(performance_summary, use_container_width=True)

        st.subheader("Latest Prediction Ranking")

        st.markdown(
            """
            This table is based on the **machine learning model's predicted probability** of
            outperforming the BIST100 over the next 20 trading days. This is different from the
            historical performance ranking above.
            """
        )

        ranking_n = st.slider(
            "Number of rows to show in prediction ranking table",
            min_value=10,
            max_value=len(tickers),
            value=min(25, len(tickers)),
        )

        ranking_df = latest_rows[
            [
                "ticker",
                "date",
                "close",
                "predicted_probability",
                "prediction_label",
                "return_20d",
                "volatility_20d",
                "volume_ratio_20d",
            ]
        ].copy()

        ranking_df["predicted_probability"] = (
            ranking_df["predicted_probability"] * 100
        ).round(2)

        ranking_df = ranking_df.rename(
            columns={
                "date": "latest_date",
                "close": "latest_close",
                "predicted_probability": "predicted_probability_pct",
            }
        )

        st.dataframe(ranking_df.head(ranking_n), use_container_width=True)

        st.subheader("Data Availability by Ticker")

        availability_df = (
            df[df["ticker"].isin(comparison_tickers)]
            .groupby("ticker", as_index=False)
            .agg(
                first_available_date=("date", "min"),
                latest_available_date=("date", "max"),
                observations=("date", "count"),
            )
            .sort_values("first_available_date")
        )

        st.dataframe(availability_df, use_container_width=True)


# -------------------------------------------------------
# Tab 3: Model notes
# -------------------------------------------------------

with tab_model:
    st.subheader("Model Notes")

    st.markdown(
        """
        - **Prediction target:** whether the stock's next 20-trading-day return is higher than the BIST100's next 20-trading-day return.
        - **Model selected:** Logistic Regression, selected based on validation ROC-AUC.
        - **Alternative model tested:** Random Forest.
        - **Pipeline:** raw Excel files were uploaded to Databricks, transformed into Bronze, Silver, and Gold Delta tables, and then used for MLflow-tracked model training.
        - **Interpretation:** the model should be interpreted as a baseline predictive system, not as financial advice.
        - **Important limitation:** the dataset is based on the current BIST100 constituent list, so survivorship bias may exist.
        - **Return interpretation:** extremely large normalized returns should be reviewed carefully because long-term Turkish equity data can be affected by very low starting prices, inflation, stock splits, bonus issues, and data adjustment conventions.
        """
    )

    st.subheader("Current Model Features")
    st.dataframe(pd.DataFrame({"feature": feature_cols}), use_container_width=True)

    st.subheader("Latest Prediction Dataset")
    st.dataframe(
        latest_rows[
            [
                "ticker",
                "date",
                "close",
                "predicted_probability",
                "prediction_label",
            ]
        ],
        use_container_width=True,
    )