# ui/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.database import read_data, get_market_list

# Page Configuration
st.set_page_config(page_title="Crypto Bot 2.0 Dashboard", layout="wide")
st.title("📈 Crypto Bot 2.0 Dashboard")

# --- Sidebar for Market and Timeframe Selection ---
st.sidebar.header("Controls")
# Dynamically get markets from the DB, with a fallback list
available_markets = get_market_list()
if not available_markets:
    st.sidebar.warning("No data found in DB. Using default markets.")
    available_markets = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]

market = st.sidebar.selectbox("Select Market", available_markets)
timeframe = st.sidebar.selectbox("Select Timeframe", ["15-Min", "1-Hour", "4-Hour"])

# --- Load Data ---
table_suffix_map = {"15-Min": "", "1-Hour": "_1H", "4-Hour": "_4H"}
table_suffix = table_suffix_map[timeframe]
data_key = f"{market}{table_suffix}"

df = read_data(data_key)

# --- Main Page Display ---
if not df.empty and 'f7' in df.columns:
    st.header(f"Analysis for {market} ({timeframe})")

    # --- Create a figure with a secondary y-axis ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Plot 1: Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df['started_at'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name=f"{market} Price"
    ), row=1, col=1)

    # Plot 2: Formula 7 Oscillator
    fig.add_trace(go.Scatter(
        x=df['started_at'], y=df['f7'], mode='lines', name='Formula 7',
        line=dict(color='cyan', width=1)
    ), row=2, col=1)

    # Add zero line for reference
    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray")

    # Add signal markers to the oscillator plot
    buy_signals = df[df['f7_signal'] == 1]
    sell_signals = df[df['f7_signal'] == -1]
    fig.add_trace(go.Scatter(
        x=buy_signals['started_at'], y=buy_signals['f7'], mode='markers', name='Buy Signal',
        marker=dict(color='lime', size=8, symbol='triangle-up')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sell_signals['started_at'], y=sell_signals['f7'], mode='markers', name='Sell Signal',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ), row=2, col=1)

    fig.update_layout(
        title=f"{market} Price and Formula 7 Indicator",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Formula 7", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Key Metrics Display
    st.subheader("Latest Metrics")
    latest = df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close Price", f"${latest['close']:,.2f}")
    col2.metric("Sentiment (CoinGecko)", f"{latest['sentiment']:.2f}% Up-votes")
    col3.metric("20-Period Volatility", f"{latest['volatility']:.5f}")

    # Raw Data Table
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(100))

elif df.empty:
    st.error(f"No data available for {market} ({timeframe}). Please run the data pipeline first.")
else:
    st.warning("Data found, but 'f7' column is missing. Please re-run the data pipeline to calculate the indicator.")