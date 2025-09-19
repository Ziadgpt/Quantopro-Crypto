# ui/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import all our celery tasks
from tasks import run_pipeline_task, train_all_models_task, run_tuning_task, run_backtest_task
from data.database import read_data, get_market_list
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Crypto Bot 2.0 Dashboard", layout="wide")

st.title("Crypto Bot 2.0 Dashboard")

# --- Sidebar Control Panel ---
st.sidebar.title("🛠️ Control Panel")
st.sidebar.info("Tasks are run in the background. Check your Celery worker terminal for progress.")

if st.sidebar.button("Run Data Pipeline"):
    task = run_pipeline_task.delay()
    st.sidebar.success(f"✅ Pipeline task started! (ID: {task.id})")

if st.sidebar.button("Train All Models"):
    task = train_all_models_task.delay()
    st.sidebar.success(f"✅ Model training task started! (ID: {task.id})")

if st.sidebar.button("Tune Synthesizer Model"):
    task = run_tuning_task.delay()
    st.sidebar.success(f"✅ Hyperparameter tuning task started! (ID: {task.id})")

if st.sidebar.button("Run Backtest"):
    task = run_backtest_task.delay()
    st.sidebar.success(f"✅ Backtest task started! (ID: {task.id})")

# --- Main Page Content ---
markets = get_market_list()

if not markets:
    st.error("No data found in the database. Run the 'Data Pipeline' from the Control Panel.")
else:
    # (The rest of the dashboard display code remains the same)
    market = st.selectbox("Select Market", sorted(markets))
    if market:
        df = read_data(market)
        if not df.empty:
            st.subheader(f"{market} 15-Min Data")
            fig = go.Figure(data=[
                go.Candlestick(x=df['started_at'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=f"{market} Price")
            ])
            fig.update_layout(title=f"{market} Candlestick Chart", xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Latest Metrics")
            latest = df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Latest Close Price", value=f"${latest['close']:.2f}")
            with col2:
                sentiment_val = latest.get('sentiment', 'N/A')
                st.metric(label="Sentiment", value=f"{sentiment_val:.2f}%" if isinstance(sentiment_val, (int, float)) else "N/A")
            with col3:
                st.metric(label="Volatility", value=f"{latest['volatility']:.4f}")

            st.subheader("Raw Data Preview")
            st.dataframe(df.tail(10))
        else:
            st.error(f"No data available for {market}")