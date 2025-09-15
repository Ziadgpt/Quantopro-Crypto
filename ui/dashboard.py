# ui/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.database import read_data, get_market_list
import logging

# It's good practice to have a logger in UI code too for debugging.
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Crypto Bot 2.0 Dashboard", layout="wide")

st.title("Crypto Bot 2.0 Dashboard")

# Dynamically get the list of markets from the database
try:
    markets = get_market_list()
except Exception as e:
    logger.error(f"Could not fetch market list from database: {e}")
    markets = []

if not markets:
    st.error("No data found in the database. Please run the data pipeline first.")
else:
    market = st.selectbox("Select Market", sorted(markets))

    if market:
        # read_data now returns a DataFrame with 'started_at' as a datetime object
        df = read_data(market)

        if not df.empty:
            st.subheader(f"{market} 15-Min Data")

            # Create the candlestick chart
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['started_at'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=f"{market} Price"
                )
            ])
            fig.update_layout(
                title=f"{market} Candlestick Chart",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
                template="plotly_dark"  # A minor visual improvement
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Latest Metrics")
            # Use .iloc[-1] to get the last row for the latest data
            latest = df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Latest Close Price", value=f"${latest['close']:.2f}")
            with col2:
                # Ensure sentiment exists and is a number before formatting
                sentiment_val = latest.get('sentiment', 'N/A')
                st.metric(label="Sentiment", value=f"{sentiment_val:.2f}%" if isinstance(sentiment_val, (int, float)) else "N/A")
            with col3:
                st.metric(label="Volatility", value=f"{latest['volatility']:.4f}")

            # Display raw data (last 10 rows for a cleaner view)
            st.subheader("Raw Data Preview")
            st.dataframe(df.tail(10))
        else:
            st.error(f"No data available for {market}")