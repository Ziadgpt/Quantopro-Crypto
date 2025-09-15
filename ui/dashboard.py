# ui/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.database import read_data

st.set_page_config(page_title="Crypto Bot 2.0 Dashboard", layout="wide")

st.title("Crypto Bot 2.0 Dashboard")
markets = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]
market = st.selectbox("Select Market", markets)

df = read_data(market)
if not df.empty:
    st.subheader(f"{market} 15-Min Data")
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
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Metrics")
    latest = df.iloc[-1]
    st.write(f"Sentiment: {latest['sentiment']}%")
    st.write(f"Volatility: {latest['volatility']:.4f}")
    st.write(f"Latest Close: ${latest['close']:.2f}")
else:
    st.error(f"No data available for {market}")

# Display raw data
st.subheader("Raw Data")
st.dataframe(df.tail())