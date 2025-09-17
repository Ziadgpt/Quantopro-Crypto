# ui/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.database import read_data, get_market_list

# Page Configuration
st.set_page_config(page_title="Crypto Bot 2.0 Dashboard", layout="wide")
st.title("📈 Crypto Bot 2.0 Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
available_markets = get_market_list() or ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]
market = st.sidebar.selectbox("Select Market", available_markets)
timeframe = st.sidebar.selectbox("Select Timeframe", ["15-Min", "1-Hour", "4-Hour"])

# --- Load Data ---
table_suffix_map = {"15-Min": "", "1-Hour": "_1H", "4-Hour": "_4H"}
table_suffix = table_suffix_map[timeframe]
data_key = f"{market}{table_suffix}"
df = read_data(data_key)

if df.empty:
    st.error(
        f"No data available for {market} ({timeframe}). Please run the pipeline first using `python main.py --pipeline`.")
else:
    # --- Main Chart with HMM Regimes ---
    st.header(f"Price Action & Market Regimes for {market}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Plot 1: Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df['started_at'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name=f"{market} Price"
    ), row=1, col=1)

    # Add LSTM Forecast if available
    if 'lstm_forecast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['started_at'], y=df['lstm_forecast'], mode='lines',
            name='LSTM Forecast', line=dict(color='orange', dash='dot')
        ), row=1, col=1)

    # Add HMM Regime backgrounds if available
    if 'hmm_regime' in df.columns:
        colors = ['rgba(0,255,0,0.1)', 'rgba(255,0,0,0.1)', 'rgba(0,0,255,0.1)', 'rgba(255,255,0,0.1)']
        for regime in df['hmm_regime'].unique():
            fig.add_vrect(
                x0=df[df['hmm_regime'] == regime]['started_at'].min(),
                x1=df[df['hmm_regime'] == regime]['started_at'].max(),
                fillcolor=colors[regime % len(colors)],
                layer="below", line_width=0,
                annotation_text=f"Regime {regime}", annotation_position="top left"
            )

    # Plot 2: Synthesizer Conviction Score
    if 'score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['started_at'], y=df['score'], mode='lines',
            name='Conviction Score', line=dict(color='cyan')
        ), row=2, col=1)
        fig.add_hline(y=0.7, row=2, col=1, line_dash="dash", line_color="lime", annotation_text="Buy Threshold")
        fig.add_hline(y=0.3, row=2, col=1, line_dash="dash", line_color="red", annotation_text="Sell Threshold")

    fig.update_layout(
        title_text=f"{market} Analysis ({timeframe})",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Conviction Score", range=[0, 1], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # --- Key Metrics ---
    st.header("Latest Data Points")
    latest = df.iloc[-1]

    cols = st.columns(5)
    cols[0].metric("Latest Close", f"${latest.get('close', 0):,.2f}")
    cols[1].metric("HMM Regime", f"{int(latest.get('hmm_regime', 'N/A'))}")
    cols[2].metric("Conviction Score", f"{latest.get('score', 0):.2%}")
    cols[3].metric("BTC Correlation", f"{latest.get('rolling_corr_btc', 0):.2f}")
    cols[4].metric("Funding Rate", f"{latest.get('funding_rate', 0):.6f}")

    # --- Raw Data Expander ---
    with st.expander("View Full Enriched Data"):
        st.dataframe(df)